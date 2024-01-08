from pymongo import MongoClient
import datetime
import pprint
import os

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

class MongoLink():
    """handles the connection itself to the database"""
    def __init__( self, mongo_client: MongoClient = None, db: object = None, collection: object = None ) -> None:
        self.client = mongo_client
        self.db = db
        self.collection = collection
        self.col = None
        pass

    def use_db( self, new_db_name: str ):
        """Updates currently accessed database"""
        self.db = self.client.get_database( new_db_name )
        pass

    def use_collection( self, new_collection_name: str ):
        """Updates currently accessed collection"""
        self.collection = self.db.get_collection( new_collection_name )
        pass

    def upload_doc( self, doc: ImageDocument ):
        """Uploads the designated document to the currently accessed collection"""
        self.collection.insert_one( doc.data )
        pass
    def upload_batch( self, doc_list: list ):
        """Uploads a batch of documents to the currently accessed collection"""
        dict_list = [doc.data for doc in doc_list]
        self.collection.insert_many( dict_list )
        pass

    def query(self):
        """Accesses MongoDB and retrieves some entire collection"""
        posts = self.collection.find()
        return posts
    
    def drop(self):
        print("Oops! Ran out of time implementing")
    
def load_doc( img_path: str, img_desc_path: str ) -> ImageDocument:
    """Loads an image and its description from file. Returns a document object"""
    with open( img_path, 'rb' ) as img_file:
        img_bin_data = img_file.read()
    with open( img_desc_path, 'r' ) as desc_file:
        img_desc = desc_file.read().split("\n")
    doc = ImageDocument( image_bin = img_bin_data, desc = img_desc )
    return doc

def create_doc( img_data: object, img_desc: str ) -> ImageDocument:
    """ Creates an image document with given parameters"""
    return ImageDocument( image_bin = img_data, desc = img_desc )

def load_descriptions( root_dir: str, desc_dir: str ) -> dict:
    """
    Reads from a directory of description files. Returns a mapping
    of each description filename with the file contents.
    
    ex:
    dict["desc_1"] = "The bird is nice and fluffy"
    
    """
    desc_dict = {}
    desc_dir = os.path.join( root_dir, desc_dir)
    for root, dirs, files in os.walk(desc_dir):
        for filename in files:
            filepath = os.path.join( root, filename )
            with open(filepath, 'r') as f:
                file_contents =  f.read()
                # Change key name as needed
                key = filename[:3]
                desc_dict[key] = file_contents
    return desc_dict

def main():
    print("Don't run mongo.py as main. There is nothing special here")
    # Constants and variables
    HOSTNAME = "localhost"
    PORT_NUMBER = 27017
    INITIAL_DB_NAME = "cs580"
    INITIAL_COLLECTION_NAME = "test"
    client = MongoClient( HOSTNAME, PORT_NUMBER )
    db = client.get_database( INITIAL_DB_NAME )
    collection = db.get_collection( INITIAL_COLLECTION_NAME )

    ml = MongoLink( mongo_client = client, db = db, collection = collection )
    # test_doc = load_doc('flowers/jpg/image_00001.jpg', None, 'flowers/text_c10/class_00077/image_00001.txt')
    # ml.upload_doc( test_doc )

    # pprint.pprint(db)
    posts = collection.find({"name":"yooo"})
    posts = db.asdf.find()
    for post in posts:
        pprint.pprint(post)

if __name__ == "__main__":
    main()