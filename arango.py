from pyArango.connection import *

class ImageDocument():
    """Container class that holds a document used by MongoDB"""
    def __init__( self, parent_dataset_name: str = "", image_bin: object = None, dataset_name: str = "", desc: str = "" ) -> None:
        # ID is automatically added
        self.data = {
            "description": desc,
            "image": image_bin,
            "parent_dataset": parent_dataset_name,
            "dataset": dataset_name
        }
        pass

    def label_parent( self, parent_dataset_name ):
        self.data["parent_dataset"] = parent_dataset_name
        pass

    def label_dataset( self, dataset_name ):
        self.data["dataset"] = dataset_name
        pass

class ArangoLink():
    def __init__( self, username: str, password: str, url: str = "http://127.0.0.1:8529", db: str = "experiments", col: str = "test"):
        # Connect to arango
        self.conn = Connection(username=username, password=password, arangoURL=url)
        # Connect to a database
        if not self.conn.hasDatabase(db):
            #check if exists
            self.conn.createDatabase(name=db) 
        self.db = self.conn[db]

        if not self.db.hasCollection(col):
            self.db.createCollection(name=col)
        self.collection = self.db[col]
        self.col = col
        pass       
    def create_doc( self, img_data: object, img_desc: str ) -> ImageDocument:
        """ Creates an image document with given parameters"""
        return ImageDocument( image_bin = img_data, desc = img_desc )

    def upload_doc( self, doc ):
        """Uploads the designated document to the currently accessed collection"""
        arango_doc = self.collection.createDocument()
        arango_doc.set(doc.data)
        arango_doc.save()
        pass

    def query(self, query=None):
        the_query = query
        if not query:
            the_query = f"""
            FOR doc IN {self.col}
                RETURN doc
            """
            result = self.db.AQLQuery(the_query, rawResults=True)

        return result
    
    def drop(self):
        print("oops! Ran out of time implementing")




def main():
    print("Don't run this script as main. There is nothing here.")

    HOSTNAME = "localhost"
    PORT_NUMBER = 8529
    INITIAL_DB_NAME = "cs580"
    INITIAL_COLLECTION_NAME = "butterflies"
    a = ArangoLink(username="root",password="1")
    print(a.col, a.db)
    # Connect to the created database
    # db = connection[db_name]
    # print(f"Connected to database: {db_name}")

    # # Create a new collection in the database
    # collection_name = "your_collection"
    # if not db.hasCollection(collection_name):
    #     new_collection = db.createCollection(name=collection_name)

    # # Connect to the created collection
    # collection = db[collection_name]
    # print(f"Connected to collection: {collection_name}")
if __name__ == "__main__":
    main()

    

