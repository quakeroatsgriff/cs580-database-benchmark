import argparse
import mongo as m
import arango as a
import postgres as p
from generator import ImageGenerator

import time as t
import pandas as pd
import io
import bson
import os
import sys
from dotenv import load_dotenv
load_dotenv()

# USERNAME = str(os.getenv('USERNAME'))
# PASSWORD = str(os.getenv('PASSWORD'))
USERNAME = "root"
PASSWORD = "1"


class GeneratorConnector():
    """This houses all the parameters and stuff for the image generator."""
    def __init__(self, IMG_HEIGHT = 32, IMG_WIDTH = 32, TOTAL_NUM_IMAGES = 10, CHECKPOINT_NUM = 10, BATCH_SIZE = 1, NUM_INFERENCES_STEPS = 10, prompts = [ "test image" ]):
        self.TOTAL_NUM_IMAGES = TOTAL_NUM_IMAGES
        self.CHECKPOINT_NUM = CHECKPOINT_NUM
        self.IMG_HEIGHT = IMG_HEIGHT
        self.IMG_WIDTH = IMG_WIDTH
        self.BATCH_SIZE = BATCH_SIZE
        self.NUM_INFERENCES_STEPS = NUM_INFERENCES_STEPS    
        self.prompts = prompts
        self.ig = ImageGenerator(img_height=IMG_HEIGHT, img_width=IMG_WIDTH,
                        batch_size=BATCH_SIZE, num_inferences_steps=NUM_INFERENCES_STEPS)
        pass
    def generate(self, db_name, db_link):
        """This generates images and inserts them into a given database. Also takes time measurement."""
        total_runtime = t.time()*1000
        av_time = 0.0

        run_num = 0
        run_parent_dir = "./runs"
        run_path = f"./runs/run{run_num}/"

        if not os.path.exists(run_parent_dir):
            os.mkdir(run_parent_dir)

        dir_list = []
        for root, dirs, files in os.walk("./runs/"):
                for dir_name in dirs:
                    dir_list.append(dir_name)
        while True:
            if os.path.exists(run_path):
                run_num += 1
                run_path = f"./runs/run{run_num}/"
            else:
                run_path = f"./runs/run{run_num}/"
                break

        os.mkdir(run_path)
        print(f"Starting run {run_num}")

        df = pd.DataFrame( columns = ['db','collection_name', 'time', 'prompt'] )

        print("Generating images now")

        for i in range(0,self.TOTAL_NUM_IMAGES):
            self.ig.new_seed()
            # Generate the new images!
            images = self.ig.generate_images(self.prompts)
            for j, (image, prompt) in enumerate(zip(images, self.prompts)):
                # Convert PNG To byte array
                image_bytes = io.BytesIO()
                image.save(image_bytes, format='JPEG')

                # Create a BSON document for binary image format
                bson_data = bson.Binary(image_bytes.getvalue())
                # Determine what to do based on database
                doc = None
                col_name = db_link.col
                if db_name == "mongo":
                    doc = m.create_doc( img_data = bson_data, img_desc = prompt )
                    start_time = t.time()*1000
                    # Do timing here
                    db_link.upload_doc(doc)
                elif db_name == "arango":
                    doc = db_link.create_doc( img_data = bson_data, img_desc = prompt )
                    start_time = t.time()*1000
                    # Do timing here
                    db_link.upload_doc(doc)
                elif db_name == "postgres":
                    doc = db_link.create_doc( img_data = bson_data, img_desc = prompt )
                    start_time = t.time()*1000
                    # Do timing here
                    db_link.upload_doc(doc)

                total_time = t.time()*1000 - start_time #end - start time                
                av_time += total_time

                print(f'Uploaded pic [{i}] in  {total_time} ms')
                new_df_row = {
                    'prompt': prompt,
                    'time': total_time,
                    'db': "mongo",
                    'collection_name': col_name
                }
                # insert the new row at the end of the DataFrame
                df.loc[len(df)] = new_df_row
                # Checkpoint every 10 image insertions
                if i % self.CHECKPOINT_NUM == 0:
                    csv_filename = f'checkpoint_{i}.csv'
                    csv_file_path = run_path + csv_filename
                    print("New checkpoint at:", csv_file_path)
                    # Write the DataFrame to a CSV file
                    df.to_csv(csv_file_path, index=False)
                    if os.path.exists(run_path + f'checkpoint_{i-self.CHECKPOINT_NUM}.csv'):
                        os.remove(run_path + f'checkpoint_{i-self.CHECKPOINT_NUM}.csv') 


            if i == (self.TOTAL_NUM_IMAGES-1):
                # Check if there are any more entries not logged
                if not os.path.exists(run_path + f'checkpoint_{i}.csv'):
                    print("New checkpoint at:", run_path + f'checkpoint_{i}.csv')
                    # Write the DataFrame to a CSV file
                    df.to_csv(run_path + f'checkpoint_{i}.csv', index=False)
            
        av_time = av_time / self.TOTAL_NUM_IMAGES
        print("Average insertion time:",av_time,"ms")

        total_runtime = t.time()*1000 - total_runtime
        print("total runtime:", total_runtime,"ms")
        print("TOTAL_NUM_IMAGES:",self.TOTAL_NUM_IMAGES)
        print("IMG_HEIGHT:",self.IMG_HEIGHT,"IMG_WIDTH:",self.IMG_WIDTH)
        print("BATCH_SIZE:",self.BATCH_SIZE)
        print("NUM_INFERENCES_STEPS:",self.NUM_INFERENCES_STEPS)
        print("prompts:",self.prompts)
        
class DatabaseConnector():
    """This establishes a connection link with some database"""
    def __init__(self, db_name):    
        self.db_name = db_name
        HOSTNAME = "localhost"
        PORT_NUMBER = 27017
        INITIAL_DB_NAME = "experiments"
        INITIAL_COLLECTION_NAME = "demo"
        # Identify which DB to use
        if self.db_name == "mongo":
            client = m.MongoClient( HOSTNAME, PORT_NUMBER )
            db = client.get_database( INITIAL_DB_NAME )
            collection = db.get_collection( INITIAL_COLLECTION_NAME )
            self.db_link = m.MongoLink( mongo_client = client, db = db, collection = collection )
            self.db_link.col = INITIAL_COLLECTION_NAME
            # Done with Mongo link set up

        elif self.db_name == "arango":
            self.db_link = a.ArangoLink(username=USERNAME,password=PASSWORD,col=INITIAL_COLLECTION_NAME, db=INITIAL_DB_NAME)
            # Done with Arango link set up

        elif self.db_name == "postgres":
            self.db_link = p.PostgresLink(username="griffen",password="griffen", hostname=HOSTNAME, col=INITIAL_COLLECTION_NAME, db=INITIAL_DB_NAME)
            # TODO
            # self.db_link = p.PostgresLink(username=USERNAME,password=USERNAME,col=INITIAL_COLLECTION_NAME, db=INITIAL_DB_NAME)
            # Done with Postgres link set up

            




def main():
    parser = argparse.ArgumentParser( description = 'Options to indicate what database to use and what operation to perform' )    
    parser.add_argument( '--db', choices = ['arango', 'mongo', 'postgres'], type = str, help = 'The database to interact with' )
    parser.add_argument( '--op', choices = ['query', 'drop', 'generate'], type = str, help = 'Which database operation to perform' )
    
    args = parser.parse_args()
    print( 'Arguments:', args )
    # Establish connection with DB
    db_connector = DatabaseConnector(db_name=args.db)
    # Can modify parameters of Generator Connector as desired
    gen_connector = GeneratorConnector()
    # gen_connector = GeneratorConnector(IMG_HEIGHT=10,IMG_WIDTH=10,NUM_INFERENCES_STEPS=5)
    
    if args.op == "generate":
        gen_connector.generate(db_link=db_connector.db_link, db_name=db_connector.db_name)
        pass
    elif args.op == "query":
        print( db_connector.db_link.query() )
        pass
    elif args.op == "drop":
        db_connector.db_link.drop()
        pass
    pass

if __name__ == "__main__":
    main()
else:
    print(f"Please run main.py as the main python file.")
    sys.exit(1)