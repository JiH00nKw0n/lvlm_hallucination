sudo apt update
sudo apt install awscli -y
# Install awscli if you don't have it (https://aws.amazon.com/cli/)
# Download Open Images packed files
aws s3 --no-sign-request --endpoint-url https://s3.amazonaws.com cp s3://open-images-dataset/tar/train_0.tar.gz .
aws s3 --no-sign-request --endpoint-url https://s3.amazonaws.com cp s3://open-images-dataset/tar/train_1.tar.gz .

# Create folder for extracted images
mkdir openimage_source_images

# Extract the tar files
tar -xvzf train_0.tar.gz -C openimage_source_images
tar -xvzf train_1.tar.gz -C openimage_source_images

# Download metadata CSV (ImageID ↔ OriginalURL mapping)
wget https://storage.googleapis.com/openimages/2018_04/train/train-images-boxable-with-rotation.csv

# Map urls to local paths
python map_openimage_url_to_local.py #please modify variable is_multi_turn and file paths as needed