import psycopg2

class ImageDocument():
    """Container class that holds a document used by MongoDB"""
    def __init__( self, parent_dataset_name: str = "", image_bin: object = None, dataset_name: str = "", desc: str = "" ) -> None:
        # ID is automatically added
        self.data = tuple([image_bin,desc])
        pass

    def label_parent( self, parent_dataset_name ):
        self.data["parent_dataset"] = parent_dataset_name
        pass

    def label_dataset( self, dataset_name ):
        self.data["dataset"] = dataset_name
        pass

class PostgresLink():
    def __init__( self, username: str, password: str, hostname: str = "localhost", db: str = "experiments", col: str = "demo"):
        # Connect to postgres
        self.conn = psycopg2.connect(
            host=hostname,
            database=db,
            user=username,
            password=password)
        
        # Execute a CREATE TABLE query 
        create_table_query = f'''
            CREATE TABLE IF NOT EXISTS {col} (
            image_desc VARCHAR,
            image_data BYTEA
        );
        '''
        self.col = col
        self.cur = self.conn.cursor()
        self.cur.execute(create_table_query)
        # Commit the changes to the database
        self.conn.commit()
        # print(self.conn)
        pass  
         
    def create_doc( self, img_data: object, img_desc: str ) -> ImageDocument:
        """ Creates an image document with given parameters"""
        return ImageDocument( image_bin = img_data, desc = img_desc )

    def upload_doc( self, doc ):
        """Uploads the designated document to the currently accessed collection"""
        insert_query = """
            INSERT INTO img (image_desc, image_data)
            VALUES (%s, %s);
        """
        self.cur.execute(insert_query, doc.data)
        self.conn.commit()
        pass
    
    def query( self, query = "SELECT * FROM demo" ):
        """Does a query. Takes in a query string and executes it."""
        self.cur.execute(query)
        rows = self.cur.fetchall()

        # print(rows[0])
        return rows
    
    def drop(self):
        "Drops the current table"
        query = f"""
            DROP TABLE IF EXISTS {self.col}
        """
        self.cur.execute(query)
        print("Done!")
        pass





def main():
    HOSTNAME = "localhost"
    PORT_NUMBER = 8529
    INITIAL_DB_NAME = "cs580"
    INITIAL_COLLECTION_NAME = "butterflies"
    pl = PostgresLink(username="griffen",password="griffen")
    pl.query()
if __name__ == "__main__":
    main()

    



