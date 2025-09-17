import psycopg2
from psycopg2 import Error
from sqlalchemy import URL, create_engine
import os
from dotenv import load_dotenv
load_dotenv()

def get_engine():
    """Cria e retorna um objeto de conexão com o PostgreSQL usando variáveis de ambiente."""
    try:
            user=os.getenv("DB_USER")
            pwd=os.getenv("DB_PASSWORD")
            hostname=os.getenv("DB_HOST")
            port=os.getenv("DB_PORT")
            db=os.getenv("DB_NAME")
            url_object = URL.create(
                "postgresql+psycopg2",
                username=user,
                password=pwd,
                host=hostname,
                database=db
            )
            engine = create_engine(url_object)
            
            return engine

    except (Exception, Error) as error:
        print(f"Erro ao gerar engine: {error}")
        return None

def close_db_connection(conn, cursor=None):
    """Fecha a conexão e o cursor, se existirem."""
    if cursor:
        cursor.close()
    if conn:
        conn.close()
        print("Conexão com o PostgreSQL fechada.")