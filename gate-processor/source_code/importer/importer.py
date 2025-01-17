import os
import mysql.connector
import cv2
import imagezmq
import zmq
import json
from time import sleep
import numpy as np
from datetime import datetime, timedelta


# Database connection details
db_config = {
    'user': 'root',
    'password': 'root',
    'host': 'gate-processor-mysql',
    'database': 'gateopener'
}



# Connect to the database
print("[*] Waiting 15 secs before starting...")
sleep(15)
db_connection = mysql.connector.connect(**db_config)
cursor = db_connection.cursor()



# Create tables if they don't exist
create_table_queries = [
    """
    CREATE TABLE IF NOT EXISTS NumberplatePictures (
        id INT AUTO_INCREMENT PRIMARY KEY,
        filename VARCHAR(255) NOT NULL UNIQUE,
        image MEDIUMBLOB NOT NULL,
        INDEX idx_filename (filename)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS Analyzed_YOLOV10XV2 (
        ID INT AUTO_INCREMENT PRIMARY KEY,
        RecognizedText VARCHAR(50) NOT NULL,
        Annotations TEXT NOT NULL,
        INDEX idx_recognizedtext (RecognizedText)
    )
    """
]
for query in create_table_queries:
    cursor.execute(query)




# Initialize ImageSender and ImageHub
NEURAL_NP_READER_HOST_PORT = 'tcp://neural-np-ocr-best:5555'
senderNpReader = imagezmq.ImageSender(connect_to=NEURAL_NP_READER_HOST_PORT)



# Function to insert an image into the database and return the inserted ID
def insert_image(filename, image_data):
    insert_query = "INSERT IGNORE INTO NumberplatePictures (filename, image) VALUES (%s, %s)"
    cursor.execute(insert_query, (filename, image_data))
    db_connection.commit()
    return cursor.lastrowid


# Function to insert annotations into the database
def insert_annotations(image_id, recognized_text, annotations):
    insert_query = "INSERT INTO Analyzed_YOLOV10XV2 (ID, RecognizedText, Annotations) VALUES (%s, %s, %s)"
    cursor.execute(insert_query, (image_id, recognized_text, annotations))
    db_connection.commit()


def neuralLPReader_sendFrame(frame):
    response = senderNpReader.send_image("client", frame)
    senderNpReader.zmq_socket.setsockopt(zmq.LINGER, 0)  # prevents ZMQ hang on exit
    senderNpReader.zmq_socket.setsockopt(zmq.RCVTIMEO, 5000)  # will raise a ZMQError exception after x ms
    senderNpReader.zmq_socket.setsockopt(zmq.SNDTIMEO, 5000)  # will raise a ZMQError exception after x ms
    responseJson = json.loads(response.decode('utf-8'))
    return responseJson



def import_directory(directory):
    # Iterate through all files in the directory
    for filename in os.listdir(directory):
        if filename.endswith(".jpg") or filename.endswith(".png"):  # Adjust the file types as needed
            filepath = os.path.join(directory, filename)
            try:
                with open(filepath, 'rb') as file:
                    image_data = file.read()

                # Analyze the image before inserting
                frame = cv2.imread(filepath)
                
                # Send image to OCR service
                annotations = neuralLPReader_sendFrame(frame)
                sorted_detections = sorted(annotations, key=lambda char: char['bbox']['x1'])
                numberplateReading = ''.join(char['class'] for char in sorted_detections)


                # Insert the image and annotations into the database
                image_id = insert_image(filename, image_data)
                insert_annotations(image_id, numberplateReading, json.dumps(annotations))


                # Remove the file after processing
                os.remove(filepath)
                print(f"Imported and analyzed: {filename}")

            except OSError as e:
                print(f"Error reading file {filepath}: {e}")



def analyze_missing_pictures(batch_size=100000):
    offset = 0
    while True:
        # Fetch images that are in NumberplatePictures but not in Analyzed_YOLOV10XV2
        fetch_query = f"""
            SELECT id, filename, image FROM NumberplatePictures
            WHERE id NOT IN (SELECT ID FROM Analyzed_YOLOV10XV2)
            LIMIT {int(batch_size)} OFFSET {int(offset)}
        """
        cursor.execute(fetch_query)
        rows = cursor.fetchall()

        if not rows:
            break

        for row in rows:
            image_id, filename, image_data = row

            # Analyze the image
            image_array = np.frombuffer(image_data, dtype=np.uint8)
            frame = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            
            if frame is None:
                print(f"Error decoding image {filename}, deleting.")
                delete_query = "DELETE FROM NumberplatePictures WHERE ID = %s"
                cursor.execute(delete_query, (image_id,))
                db_connection.commit()
                continue


            # Send image to OCR service
            annotations = neuralLPReader_sendFrame(frame)
            sorted_detections = sorted(annotations, key=lambda char: char['bbox']['x1'])
            numberplateReading = ''.join(char['class'] for char in sorted_detections)

            # Insert annotations into the database
            insert_annotations(image_id, numberplateReading, json.dumps(annotations))

            print(f"Analyzed and updated: {filename}")

        offset += batch_size






def remove_empty_pictures():
    print("[*] Removing pictures that no symbols were recognized (function: remove_empty_pictures)")

    # Get the maximum ID from NumberplatePictures
    print("[*] Retrieving max ID from 'NumberplatePictures' DB table ...")
    cursor.execute("SELECT MAX(id) FROM NumberplatePictures")
    max_id_pictures = cursor.fetchone()[0] or 0  # Handle None if table is empty

    batch_size = 1000000  # Define the batch size
    total_deleted_rows = 0

    print("[*] Removing pictures from 'NumberplatePictures' DB table in batches ...")
    for start_id in range(0, max_id_pictures + 1, batch_size):
        end_id = start_id + batch_size

        cursor.execute(f'''
            DELETE NumberplatePictures
            FROM NumberplatePictures
            LEFT JOIN Analyzed_YOLOV10XV2
            ON Analyzed_YOLOV10XV2.ID = NumberplatePictures.id
            WHERE Analyzed_YOLOV10XV2.RecognizedText = ""
            AND NumberplatePictures.id >= {start_id}
            AND NumberplatePictures.id < {end_id};
        ''')
        deleted_rows = cursor.rowcount
        db_connection.commit()
        total_deleted_rows += deleted_rows
        print(f" - Deleted {deleted_rows} rows from ID {start_id} to {end_id} in 'NumberplatePictures'.")

    print(f" - Total deleted rows from 'NumberplatePictures': {total_deleted_rows}.")

    # Now, remove corresponding records from Analyzed_YOLOV10XV2
    print("[*] Removing analysis records from 'Analyzed_YOLOV10XV2' DB table ...")
    cursor.execute("SELECT MAX(ID) FROM Analyzed_YOLOV10XV2")
    max_id_analyzed = cursor.fetchone()[0] or 0  # Handle None if table is empty

    total_deleted_rows = 0
    for start_id in range(0, max_id_analyzed + 1, batch_size):
        end_id = start_id + batch_size

        cursor.execute(f'''
            DELETE Analyzed_YOLOV10XV2
            FROM Analyzed_YOLOV10XV2
            LEFT JOIN NumberplatePictures
            ON Analyzed_YOLOV10XV2.ID = NumberplatePictures.id
            WHERE NumberplatePictures.id IS NULL
            AND Analyzed_YOLOV10XV2.ID >= {start_id}
            AND Analyzed_YOLOV10XV2.ID < {end_id};
        ''')
        deleted_rows = cursor.rowcount
        db_connection.commit()
        total_deleted_rows += deleted_rows
        print(f" - Deleted {deleted_rows} rows from ID {start_id} to {end_id} in 'Analyzed_YOLOV10XV2'.")

    print(f" - Total deleted rows from 'Analyzed_YOLOV10XV2': {total_deleted_rows}.")
    print("[*] Done. (function: remove_empty_pictures)\n")






def optimize_database():
    print("[*] Optimizing database table 'NumberplatePictures' ...")
    cursor.execute(f'OPTIMIZE TABLE NumberplatePictures')
    cursor.fetchall()

    print("[*] Optimizing database table 'Analyzed_YOLOV10XV2' ...")
    cursor.execute(f'OPTIMIZE TABLE Analyzed_YOLOV10XV2')
    cursor.fetchall()

    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"[*] Purging binary logs before '{current_time}' ...")
    cursor.execute(f"PURGE BINARY LOGS BEFORE '{current_time}'")
    cursor.fetchall()






# analyze_missing_pictures()
# remove_empty_pictures()


cycle = 0
while True:
    cycle += 1

    # Execute every 5 mins
    import_directory("saved_numberplates")
    print("[*] All images have been successfully imported and analyzed.")


    # Execute every ~3 days
    if(cycle % 100 == 0):
        analyze_missing_pictures()
        remove_empty_pictures()
        # optimize_database()


    # Sleep 5 mins
    sleep(300) 
