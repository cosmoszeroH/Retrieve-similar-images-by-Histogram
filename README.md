# Retrieve-similar-images-by-Histogram

I have built a simple application to retrieve images that have similar histogram with images in database.

To run code, the below Python libraries is necessary:
```
pickle
cv2
numpy
streamlit
```

Firstly, you should calculate the histogram of images of your database.

In ```preprocess_dataset.py``` file, replacing ```HIST_PATH``` and ```IMAGE_PATH``` variables with the path to save calculated histogram and the path to your image database, respectively. 

Then, run ```python preprocess_dataset.py```.

After doing above steps, ```streamlit run api.py``` to open the application.
