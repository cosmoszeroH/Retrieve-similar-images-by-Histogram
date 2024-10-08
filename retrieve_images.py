import cv2 as cv
import numpy as np


# Calculate the similarity between input image's histogram and all the histogram in database
def calculate_similarity(image, hist_lists):
    img_hist = cv.calcHist([image], [0], None, [256], [0, 255])
    sim_list = []
    for hist_list in hist_lists:
        for hist in hist_list:
            sim = hist.T @ img_hist / (255**2)
            sim_list.append(sim)

    return np.array(sim_list).flatten()


def get_top_k_most_similar(sim_list, top_k=10):
    return np.argpartition(sim_list,-top_k)[-top_k:]


# Get the path of retrieved images
def retrieve_similar_images(image, hist_lists, top_k=10):
    sim_list = calculate_similarity(image, hist_lists)

    return get_top_k_most_similar(sim_list, top_k)
