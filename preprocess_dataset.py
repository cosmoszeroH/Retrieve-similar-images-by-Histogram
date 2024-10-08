import cv2 as cv
import os
import pickle


HIST_PATH = r'seg_hist'
IMAGE_PATH = r'seg'


# Create folder that contains histogram of dataset's image
def create_seg_folder(hist_path):
    if not os.path.exists(hist_path):
        os.mkdir(hist_path)


# Save histogram of each image
def save_hist(image_path, hist_path):
    for root, _, files in os.walk(image_path):
        if len(root.split('\\')) == 1:
            continue
        dir = root.split('\\')[-1]
        hist_list = []
        for file in files:
            image = cv.imread(os.path.join(root, file), cv.IMREAD_GRAYSCALE)
            hist = cv.calcHist([image], [0], None, [256], [0, 255])

            hist_list.append(hist)
        
        with open(f'{hist_path}\\{dir}', 'wb') as file:
                pickle.dump(hist_list, file)
            


# Load the histograms
def load_hist(hist_path):
    hist_lists = []

    for root, _, files in os.walk(hist_path):
        for file in files:
            with open(f'{root}\\{file}', 'rb') as f:
                hist = pickle.load(f)
                hist_lists.append(hist)
        
    return hist_lists


def main():
    create_seg_folder(HIST_PATH)
    save_hist(IMAGE_PATH, HIST_PATH)


if __name__ == "__main__":
    main()