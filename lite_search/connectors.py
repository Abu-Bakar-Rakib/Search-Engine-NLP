import os
import sqlite3
import pandas as pd
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Iterator, Tuple
from .schema import CanonicalProduct, SchemaValidator, SchemaProfiler

class BaseConnector(ABC):
    """Abstract interface for all product catalog data connectors."""
    
    def __init__(self, schema_map: Dict[str, str]):
        self.schema_map = schema_map
        self.validator = SchemaValidator(schema_map)

    @abstractmethod
    def load_products(self, limit: Optional[int] = None) -> List[CanonicalProduct]:
        """Loads and returns canonical products from the data source."""
        pass

    @abstractmethod
    def iter_changes(self, since: str) -> Iterator[CanonicalProduct]:
        """Yields products modified since the given timestamp or watermark."""
        pass

    @abstractmethod
    def profile_schema(self) -> Dict[str, Any]:
        """Profiles the underlying data schema, nulls, and outliers."""
        pass


class SQLiteConnector(BaseConnector):
    """Read-only SQLite connector for local development, demo catalogs, and test suites."""
    
    def __init__(self, db_path: str, table_name: str, schema_map: Dict[str, str]):
        super().__init__(schema_map)
        self.db_path = db_path
        self.table_name = table_name

    def _get_connection(self) -> sqlite3.Connection:
        if not os.path.exists(self.db_path):
            raise FileNotFoundError(f"SQLite database file not found at: {self.db_path}")
        # Open in URI read-only mode to prevent any modification
        uri = f"file:{os.path.abspath(self.db_path)}?mode=ro"
        return sqlite3.connect(uri, uri=True)

    def load_products(self, limit: Optional[int] = None) -> List[CanonicalProduct]:
        conn = self._get_connection()
        try:
            query = f"SELECT * FROM {self.table_name}"
            if limit:
                query += f" LIMIT {int(limit)}"
            df = pd.read_sql_query(query, conn)
            return self.validator.transform_dataframe(df)
        finally:
            conn.close()

    def iter_changes(self, since: str) -> Iterator[CanonicalProduct]:
        conn = self._get_connection()
        try:
            time_col = self.schema_map.get("updated_at", "updated_at")
            query = f"SELECT * FROM {self.table_name} WHERE {time_col} > ? ORDER BY {time_col} ASC"
            cursor = conn.cursor()
            cursor.execute(query, (since,))
            cols = [col[0] for col in cursor.description]
            for row in cursor.fetchall():
                row_dict = dict(zip(cols, row))
                yield self.validator.transform_row(row_dict)
        finally:
            conn.close()

    def profile_schema(self) -> Dict[str, Any]:
        conn = self._get_connection()
        try:
            df = pd.read_sql_query(f"SELECT * FROM {self.table_name} LIMIT 5000", conn)
            return SchemaProfiler.profile_dataframe(df)
        finally:
            conn.close()


class CSVConnector(BaseConnector):
    """CSV connector supporting file uploads, exports, and batch processing."""
    
    def __init__(self, csv_path: str, schema_map: Dict[str, str]):
        super().__init__(schema_map)
        self.csv_path = csv_path

    def load_products(self, limit: Optional[int] = None) -> List[CanonicalProduct]:
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"CSV file not found at: {self.csv_path}")
        df = pd.read_csv(self.csv_path, nrows=limit)
        return self.validator.transform_dataframe(df)

    def iter_changes(self, since: str) -> Iterator[CanonicalProduct]:
        # For CSV, read and filter by updated_at column
        df = pd.read_csv(self.csv_path)
        time_col = self.schema_map.get("updated_at", "updated_at")
        if time_col in df.columns:
            filtered = df[df[time_col] > since]
            for _, row in filtered.iterrows():
                yield self.validator.transform_row(row.to_dict())
        else:
            for _, row in df.iterrows():
                yield self.validator.transform_row(row.to_dict())

    def profile_schema(self) -> Dict[str, Any]:
        df = pd.read_csv(self.csv_path, nrows=5000)
        return SchemaProfiler.profile_dataframe(df)


class SQLAlchemyConnector(BaseConnector):
    """Production-grade SQLAlchemy connector for PostgreSQL, MySQL, MSSQL, Oracle, etc."""
    
    def __init__(self, connection_string: str, table_name: str, schema_map: Dict[str, str]):
        super().__init__(schema_map)
        self.connection_string = connection_string
        self.table_name = table_name

    def _get_engine(self):
        from sqlalchemy import create_engine
        return create_engine(self.connection_string, execution_options={"readonly": True})

    def load_products(self, limit: Optional[int] = None) -> List[CanonicalProduct]:
        engine = self._get_engine()
        query = f"SELECT * FROM {self.table_name}"
        if limit:
            query += f" LIMIT {int(limit)}"
        with engine.connect() as conn:
            df = pd.read_sql(query, conn)
        return self.validator.transform_dataframe(df)

    def iter_changes(self, since: str) -> Iterator[CanonicalProduct]:
        engine = self._get_engine()
        time_col = self.schema_map.get("updated_at", "updated_at")
        query = f"SELECT * FROM {self.table_name} WHERE {time_col} > :since"
        with engine.connect() as conn:
            result = conn.execute(query, {"since": since})
            cols = result.keys()
            for row in result:
                yield self.validator.transform_row(dict(zip(cols, row)))

    def profile_schema(self) -> Dict[str, Any]:
        engine = self._get_engine()
        with engine.connect() as conn:
            df = pd.read_sql(f"SELECT * FROM {self.table_name} LIMIT 5000", conn)
        return SchemaProfiler.profile_dataframe(df)


class VectorDBAdapterInterface(ABC):
    """Pluggable adapter interface for connecting to an existing company vector database (Qdrant, Milvus, Pinecone, pgvector)."""
    
    @abstractmethod
    def search_vectors(self, query_vector: List[float], top_k: int, filters: Optional[Dict[str, Any]] = None) -> List[Tuple[str, float]]:
        """Returns list of (product_id, similarity_score)."""
        pass
