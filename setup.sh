# Setup script. A little heavy handed on the data downloads.

set -e

# Install python packages
pip install -r requirements.txt

# Install nltk data
python -m nltk.downloader all

# Unzip the data
unzip final-data.zip

# Download embeddings
pushd data
tar -xzf embeddings.tar.gz
popd
