# -------------------------
# needed imports
# ---------------------------

import time
import cv2
import os
import scipy
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt

from openpyxl import Workbook
from openpyxl.styles import Font

from sklearn import mixture
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.cluster import MiniBatchKMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# -----------------------------------------------------------------------------------------
# ---------------- SUPPORTING FUNCTIONS GO HERE -------------------------------------------
# -----------------------------------------------------------------------------------------

# load images from folder and return two dictionaries (train-test) that hold all images category by category
def load_images_from_folder(folder_name, image_size, test_set_size):
    train_img = {}
    test_img = {}
    print("Reading Dataset\n------------------------------")
    for each_category in os.listdir(folder_name):
        category = []
        path = folder_name + "/" + each_category
        print('\tCategory..', each_category, '\tparsing images..')
        for image in os.listdir(path):
            img = cv2.imread(path + "/" + image)
            if img is not None:
                # grayscale it and resize it to the given dimensions
                img = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (image_size[0], image_size[1]))
                category.append(img)
        # split images to train and test in the desired ratio
        cat_train, cat_test = train_test_split(category, test_size=test_set_size, random_state=3)
        train_img[each_category] = cat_train
        test_img[each_category] = cat_test
    return train_img, test_img


# Creates descriptors using : ORB, SIFT, SURF or BRIEF
# Takes one parameter that is images dictionary and the name of the detector we will use e.g. "ORB"
# Return an array whose first index holds the decriptor_list without an order
# And the second index holds the sift_vectors dictionary which holds the descriptors but this is seperated class by class
def detector_features(images, detector_to_use):
    detector_vectors = {}
    descriptor_list = []
    print('Detector ', detector_to_use[0], "\tstart detecting\n---------")
    t = time.time()
    for name_of_category, available_images in images.items():
        features = []
        for img in available_images:
            if detector_to_use[0] == "BRIEF":
                star = cv2.xfeatures2d.StarDetector_create()
                star_kp = star.detect(img, None)
                # compute the descriptors with BRIEF
                kp, des = detector_to_use[1].compute(img, star_kp)
            elif detector_to_use[0] == "SIFT" or detector_to_use[0] == "SURF" or detector_to_use[0] == "ORB":
                kp, des = detector_to_use[1].detectAndCompute(img, None)
            else:
                print("ERROR! DETECTOR NOT SPECIFIED")
                exit(2)
            # in case one image doesn't have key-points don't extend the list
            if len(kp) != 0:
                descriptor_list.extend(des)
                features.append(des)
        detector_vectors[name_of_category] = features
        print('\tfinished detecting points and calculating features for category ', name_of_category)
    detector_run_time = time.time() - t
    print('Computing with', detector_to_use[0], ' finished successfully in', detector_run_time, 'seconds\n--------')
    return [descriptor_list, detector_vectors]  # ONE output as a list

# This function has 3 parameters 1. clustering model to use - either K-Means or GMM
# 2. descriptors list(unordered 1d array) and 3. number of clusters in case
# the algorithm is K-Means(in case of GMM that parameters is omitted
# Returns an array that holds central points.
def visual_words_creation(clustering_model, descriptor_list, no_clusters=None):
    print(' . calculating central points for the existing feature values.')
    print('no of clusters : ', no_clusters, '\n')
    t = time.time()
    batch_size = np.ceil(descriptor_list.__len__() / 50).astype('int')
    if clustering_model == "K-Means":
        print('K-Means\n----------------------------')
        model = MiniBatchKMeans(n_clusters=no_clusters, batch_size=batch_size, verbose=0)
        model.fit(descriptor_list)
        vis_words = model.cluster_centers_
    elif clustering_model == "GMM":
        print('GMM\n----------------------------')
        # bandwidth = estimate_bandwidth(descriptor_list, quantile=0.2, n_samples=10000)
        # model = MeanShift(bandwidth, bin_seeding=True, n_jobs=3)
        model = mixture.GaussianMixture(n_components=10, covariance_type='full')
        model.fit(descriptor_list)
        vis_words = np.empty(shape=(model.n_components, len(descriptor_list[0])))
        # Calculating centroids for GMM since it doesn't calculate them itself
        for i in range(model.n_components):
            density = scipy.stats.multivariate_normal(cov=model.covariances_[i],
                                                      mean=model.means_[i],
                                                      allow_singular=True).logpdf(descriptor_list)
            vis_words[i, :] = descriptor_list[np.argmax(density)]
    else:
        print("ERROR! MODEL NOT SPECIFIED")
        exit(3)
    cluster_run_time = time.time() - t
    print('Clustering with', clustering_model, ' finished successfully in', cluster_run_time, 'seconds\n--------')
    return vis_words, model

# Creation of the histograms. To create our each image by a histogram. We will create a vector of k values for each
# image. For each keypoints in an image, we will find the nearest center, defined using training set
# and increase by one its value
def map_feature_vals_histogram(datafeaturesbyclass, visual_words, trained_model):
    # depending on the approach you may not need to use all inputs
    histograms_list = []
    targetclass_list = []
    no_bins_histogram = visual_words.shape[0]
    for cat_idx, feature_values in datafeaturesbyclass.items():
        for tmp_image_features in feature_values:  # yes, we check one by one the values in each image for all images
            tmp_image_histogram = np.zeros(no_bins_histogram)
            tmp_idx = list(trained_model.predict(tmp_image_features))
            clustervalue, visualwordmatchcounts = np.unique(tmp_idx, return_counts=True)
            tmp_image_histogram[clustervalue] = visualwordmatchcounts
            # do not forget to normalize the histogram values
            detectedpointsin_image = tmp_idx.__len__()
            tmp_image_histogram = tmp_image_histogram / detectedpointsin_image
            # now update the input and output corresponding lists
            histograms_list.append(tmp_image_histogram)
            targetclass_list.append(cat_idx)
    return histograms_list, targetclass_list

# function used to create a 10x10 confusion matrix
def show_confusion_matrix(y, y_predictions, title, true_labels):
    cm = confusion_matrix(y, y_predictions)
    plt.figure(figsize=(10, 10))
    sn.heatmap(cm, annot=True, fmt="d", cmap="YlGn")
    plt.title(title, fontsize=30)
    tick_marks = np.arange(len(true_labels))
    plt.yticks(tick_marks, true_labels, rotation=0)
    plt.xticks(tick_marks, true_labels)
    plt.ylabel('True label', fontsize=25)
    plt.xlabel('Predicted label', fontsize=25)
    plt.show()

# function used to predict labels based on the chosen classifier and saves all the results in an excel
def scores_excel_file(train_x, train_y, test_x, test_y, classifier_model, classifier_name,
                      ratio_per, detector_name, cluster_name):
    labels = np.unique(train_y)
    title = classifier_name + "-" + detector_name + "-" + cluster_name
    y_pred_train = classifier_model.predict(train_x)
    show_confusion_matrix(train_y, y_pred_train, "CM-Train-" + title, labels)
    y_pred_test = classifier_model.predict(test_x)
    show_confusion_matrix(test_y, y_pred_test, "CM-Test-" + title, labels)
    # calculate the scores
    acc_train = accuracy_score(train_y, y_pred_train)
    acc_test = accuracy_score(test_y, y_pred_test)
    pre_train = precision_score(train_y, y_pred_train, average='macro')
    pre_test = precision_score(test_y, y_pred_test, average='macro')
    rec_train = recall_score(train_y, y_pred_train, average='macro')
    rec_test = recall_score(test_y, y_pred_test, average='macro')
    f1_train = f1_score(train_y, y_pred_train, average='macro')
    f1_test = f1_score(test_y, y_pred_test, average='macro')
    data_for_xls = [[classifier_name, detector_name, cluster_name, ratio_per,
                     str(acc_train), str(pre_train), str(rec_train), str(f1_train),
                     str(acc_test), str(pre_test), str(rec_test), str(f1_test)]]
    for data in data_for_xls:
        sheet.append(data)

    print('Accuracy scores of ', classifier_name, ' classifier are:',
          'train: {:.2f}'.format(acc_train), 'and test: {:.2f}.'.format(acc_test))
    print('Precision scores of ', classifier_name, ' classifier are:',
          'train: {:.2f}'.format(pre_train), 'and test: {:.2f}.'.format(pre_test))
    print('Recall scores of ', classifier_name, ' classifier are:',
          'train: {:.2f}'.format(rec_train), 'and test: {:.2f}.'.format(rec_test))
    print('F1 scores of ', classifier_name, ' classifier are:',
          'train: {:.2f}'.format(f1_train), 'and test: {:.2f}.'.format(f1_test))
    print('')

# main function used to call all the other functions and to fit the classifiers
def bag_of_visual_words(train_images, test_images, ratio, detector, cluster_model):
    train_data_features = detector_features(train_images, detector)
    # Takes the descriptor list which is unordered one
    train_descriptor_list = train_data_features[0]
    train_BoVW_feature_vals = train_data_features[1]
    # create the central points for the histograms using k means.
    # here we use a rule of the thumb to create the expected number of cluster centers
    number_of_classes = train_images.__len__()  # retrieve num of classes from dictionary
    possible_num_centers = 10 * number_of_classes
    if cluster_model == "K-Means":
        visual_words, trained_model = visual_words_creation(cluster_model, train_descriptor_list,
                                                        possible_num_centers)
    else:
         visual_words, trained_model = visual_words_creation(cluster_model, train_descriptor_list)

    # create the train input train output format
    train_histogramslist, train_targetslist = map_feature_vals_histogram(train_BoVW_feature_vals, visual_words,
                                                                         trained_model)
    # Convert Categorical Data For Scikit-Learn
    x_train = np.stack(train_histogramslist, axis=0)
    # Create a label (category) encoder object
    labelEncoder = preprocessing.LabelEncoder()
    labelEncoder.fit(train_targetslist)
    # convert the categories from strings to names
    y_train = labelEncoder.transform(train_targetslist)

    test_data_features = detector_features(test_images, detector)
    test_BoVW_feature_vals = test_data_features[1]
    test_histogramslist, test_targetslist = map_feature_vals_histogram(test_BoVW_feature_vals, visual_words,
                                                                       trained_model)
    x_test = np.array(test_histogramslist)
    y_test = labelEncoder.transform(test_targetslist)
    train_per = str(100 - (ratio * 100)) + "%"

    knn = KNeighborsClassifier().fit(x_train, y_train)
    scores_excel_file(x_train, y_train, x_test, y_test, knn, "KNeighbors", train_per, detector[0], cluster_model)

    clf = DecisionTreeClassifier().fit(x_train, y_train)
    scores_excel_file(x_train, y_train, x_test, y_test, clf, "Decision Tree", train_per, detector[0], cluster_model)

    gnb = GaussianNB().fit(x_train, y_train)
    scores_excel_file(x_train, y_train, x_test, y_test, gnb, "GaussianNB", train_per, detector[0], cluster_model)

    svm = SVC().fit(x_train, y_train)
    scores_excel_file(x_train, y_train, x_test, y_test, svm, "SVM", train_per, detector[0], cluster_model)

# ---------------------------------------------------------------------------------

# Excel file info - headers - name of the sheet etc
headers = ['Classifier', 'FeatureExtraction', 'Clustering Detection', 'Train Data ratio',
           'Accuracy(tr)', ' Precision(tr)', 'Recall(tr)', 'F1 score(tr)',
           'Accuracy(te)', ' Precision(te)', 'Recall(te)', 'F1 score(te)']
workbook_name = 'OutputFiles/classification_scores.xls'
wb = Workbook()
sheet = wb.active
sheet.title = 'Classification Scores'
sheet.append(headers)
row = sheet.row_dimensions[1]
row.font = Font(bold=True)

images_file_path = 'monkeys/training' # path to the dataset
input_image_size = [200, 200, 3]  # define the FIXED size that CNN will have as input
test_size = [0.2, 0.4]  # size of test set : 20%, 40%
# detectors used
detect_methods = (("ORB", cv2.ORB_create()),
                  ("BRIEF", cv2.xfeatures2d.BriefDescriptorExtractor_create()),
                  ("SURF", cv2.xfeatures2d.SURF_create(400)),
                  ("SIFT", cv2.xfeatures2d.SIFT_create()))
# clustering methods used
clustering_method = ("K-Means",
                     "GMM")
t = time.time()
# load the train-test images
for ratio in test_size:
    train_set, test_set = load_images_from_folder(images_file_path, input_image_size, ratio)
    # calculate points and descriptor values per image
    for method_num in range(len(detect_methods)):
        for clust_model in clustering_method:
            bag_of_visual_words(train_set, test_set, ratio, detect_methods[method_num], clust_model)
wb.save(filename=workbook_name)
script_run_time = time.time() - t
print('Script finished running successfully after ', script_run_time, ' seconds.')
