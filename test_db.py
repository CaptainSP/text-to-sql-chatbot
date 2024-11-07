import sqlite3
import os

def test_database():
    # Print current working directory
    print("Current working directory:", os.getcwd())
    
    # Check if database file exists
    db_path = "database.db"
    print(f"Database file exists: {os.path.exists(db_path)}")
    
    try:
        # Try to connect to the database
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # List all tables in the database
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        print("\nTables in database:", tables)
        
        # Try to query the employees table
        cursor.execute("SELECT COUNT(*) FROM employees;")
        count = cursor.fetchone()[0]
        print(f"\nNumber of employees: {count}")
        
    except sqlite3.Error as e:
        print(f"Database error: {e}")
    finally:
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    test_database()