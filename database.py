
import mysql.connector

# Establecer una conexión a la base de datos MySQL
connection = mysql.connector.connect(
    host='tu_host',
    user='tu_usuario',
    password='tu_contraseña',
    database='tu_base_de_datos'
)

# Crear un cursor para ejecutar consultas SQL
cursor = connection.cursor()

# Crear una tabla para almacenar vectores faciales
create_table_query = """
CREATE TABLE IF NOT EXISTS vectores_faciales (
    id INT AUTO_INCREMENT PRIMARY KEY,
    nombre VARCHAR(255) NOT NULL,
    vector TEXT NOT NULL
)
"""

cursor.execute(create_table_query)
connection.commit()

# Insertar un vector facial en la tabla
nombre = 'Persona1'
vector = '0.1, 0.2, 0.3, ...'  # Reemplaza con el vector real
insert_query = "INSERT INTO vectores_faciales (nombre, vector) VALUES (%s, %s)"
cursor.execute(insert_query, (nombre, vector))
connection.commit()

# Recuperar un vector facial de la tabla
nombre_a_buscar = 'Persona1'
select_query = "SELECT vector FROM vectores_faciales WHERE nombre = %s"
cursor.execute(select_query, (nombre_a_buscar,))
vector_recuperado = cursor.fetchone()

if vector_recuperado:
    print(f"Vector facial de {nombre_a_buscar}: {vector_recuperado[0]}")

# Cerrar el cursor y la conexión
cursor.close()
connection.close()
