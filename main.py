"""
THe program creates recurrent neural network models (LSTM, GRU) using tensorflow keras libraries on time series
data containing a data column that must at least contain the "Close" column. The program will return R^2, Mape, and RMSE
metrics of evaluation of the models. The program saves all statistics from training into "Results/results.csv" file and also
saves the model that has achieved the highest R^2 during training so that its prediction can be visualised using pyplot.
"""
import os
import pandas as pd
import math
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, GRU, Dropout
import matplotlib.pyplot as plt
import csv
import warnings
from tensorflow import keras

warnings.filterwarnings("ignore")

#Program global variables
global selected_features, epoch_num, look_ahead_num, time_step_num, use_window_scalar, dataset_path, batch_size, net_type

global path_to_model

global load_model
load_model = False
#Global prediction scaler used to de scale prediction data by coping the limits of the local function scalar.
#This is done because prediction data is will be a 1D array and x_test data is not guaranteed to be the same dimension.
global prediction_scaler
prediction_scaler = MinMaxScaler()

#intialises the selected features as a list so that the gui can append to it.
selected_features = []

# Method takes a list of inputs and loads global based on that. This is useful when arguments are passed via
# commandline, or model loading is selected.
def load_globals(input_list):
    global selected_features, epoch_num, look_ahead_num, time_step_num, use_window_scalar, dataset_path, batch_size, net_type, load_model, path_to_model
    dataset_path = input_list['Dataset']
    selected_features = input_list['Features'].split("-")
    epoch_num = input_list['Epoch']
    look_ahead_num = input_list['LookAheadNum']
    time_step_num = input_list['TimeStepNum']
    use_window_scalar = input_list['UseWindowScalar']
    net_type = input_list['NetworkType']
    batch_size = input_list['BatchSize']
    load_model = True

#some datasets come with the latest date as the first entry this method reverses the set and saves it as csv.
def reverse_data_frame(new_csv_name, csv_to_be_reversed):
    #Copies the header of the current data.
    header = csv_to_be_reversed.columns
    #variable holds the path to the dataset directory.
    path_to_file = "Datasets/" + new_csv_name
    #using with open doesn't require the writer to be closed later.
    with open(path_to_file, 'w') as file:
        writer = csv.writer(file)
        writer.writerow(header)
        #using optional parameter increment step -1 in order to iterate backwards
        for sample in range(len(csv_to_be_reversed)-1,-1,-1):
            writer.writerow(csv_to_be_reversed.iloc[sample])

#display predictions vs actual results for the test results.
def display_results_graph(results_frame):
    plt.figure(figsize=(10, 5))
    plt.title("Close Price History")
    plt.plot(results_frame["Prediction"])
    plt.plot(results_frame["Close"])
    plt.xlabel("sample num", fontsize=18)
    plt.ylabel("Close Price USD", fontsize=18)
    plt.legend(["Actual", "Prediction"], loc="lower right")
    plt.show()

#function to read the provided csv and extract desired features from it
def read_dataset(file_path):
    #parse dates creates weird interactions with small time step datasets
    #data = pd.read_csv(file_path, parse_dates=["Date"])
    data = pd.read_csv(file_path)
    return data

#Method takes the read in dataset and returns only the columns desired for training
def filter_dataset(feature_list, data):
    # creates a new data frame containing only the desired features from the dataset
    filtered_dataset = data.filter(feature_list)
    # convert the dataframe into a numpy array
    np_filtered_array = filtered_dataset.values
    # returns filtered numpyArray which is required by Sequential models for training
    return np_filtered_array

#Method used to constrain command line input to bounded integers.
def request_input_until_within_bounds(upper, lower):
    while True:
        selected_num = input(">> ")
        if selected_num.isnumeric():
            selected_num = int(selected_num)
            if lower <= selected_num <= upper:
                break
            else:
                print("selected option out of bounds")
        else:
            print("incorrect input try again")

    return selected_num

#Method used to capture user input allowing for the creation of a new model, This method sets global values which are later used by the models
def user_ui():
    print("Make new model or load existing?")
    print("[1] New Model")
    print("[2] Load Model")
    model_list = []
    if request_input_until_within_bounds(2,1) == 2:
        results_csv = pd.read_csv("Results/results.csv")
        for model in range(len(results_csv)):
            if results_csv.loc[model]["Epoch"] == 1:
                model_list.append(results_csv.loc[model])
        models = pd.DataFrame(model_list)


        for rnn_models in range(len(models)):
            print("[" + str(rnn_models + 1) + "]\n" + str(models.iloc[rnn_models]) +"\n")
        selected_model = request_input_until_within_bounds(len(models), 1)
        selected_model = models.iloc[selected_model -1]
        load_globals(selected_model[1:9])
        global path_to_model
        path_to_model = selected_model["Model"]
    else:
        print(
            "Welcome, Please select a dataset from the list below by entering the integer corresponding to its position.")
        print(
            "Additional datasets can be added in \"root/Datasets\" directory. Ensure that the dataset contains a \"Close\" column before adding the dataset.\n")
        dataset_list = os.listdir("Datasets")
        print("Datasets:")
        for dataset in range(1, len(dataset_list) + 1):
            num = "[" + str(dataset) + "]"
            print(num, dataset_list[dataset - 1])

        selected_num = request_input_until_within_bounds(len(dataset_list), 1)
        print("Selected dataset:" + dataset_list[selected_num - 1], "\n")

        print(
            "Select features from the dataset, take care to no use features such as date or unix, ensure that close is the first selected feature")
        read_in_data = read_dataset(("Datasets/" + dataset_list[selected_num - 1]))
        for feature in range(0, len(read_in_data.columns)):
            num = "[" + str(feature + 1) + "]"
            print(num, read_in_data.columns[feature])
        print("\n[0] Exit")
        while True:
            feature = request_input_until_within_bounds(len(read_in_data.columns), 0)
            if feature == 0:
                break
            else:
                selected_features.append(read_in_data.columns[feature - 1])
        print("Currently selected features:", selected_features)

        print("Specify the number of epochs, look ahead num, and Time step window")
        print("Epoch Num")
        global epoch_num
        epoch_num = request_input_until_within_bounds(1000000, 0)
        print("Look Ahead Num")
        global look_ahead_num
        look_ahead_num = request_input_until_within_bounds(100, 0)
        print("Time Window num")
        global time_step_num
        time_step_num = request_input_until_within_bounds(10000, 0)
        print("Use scalar window?")
        print("[1] Yes")
        print("[2] No")
        global use_window_scalar
        selected_global_scaler = request_input_until_within_bounds(2, 1)
        if selected_global_scaler == 1:
            use_window_scalar = True
        else:
            use_window_scalar = False
        global dataset_path
        dataset_path = "Datasets/" + dataset_list[selected_num - 1]

        global net_type
        print("Select neural network type")
        print("[1] GRU")
        print("[2] LSTM")
        selected_net_type = request_input_until_within_bounds(2, 1)
        if selected_net_type == 1:
            net_type = "GRU"
        else:
            net_type = "LSTM"

        global batch_size
        print("Specify training batch size")
        batch_size = request_input_until_within_bounds(10000, 0)

#method used to scale and create desired shape train/test sets
def create_set(data, num_of_time_steps, look_ahead_num, scalar_window_size, is_test_data, scaler):
    #list to contain the scaled set
    scaled_data = []
    #establishes the number of windows to be scaled
    num_of_segments = math.floor(len(data) / scalar_window_size)

    for segment in range(0, num_of_segments):
        segment_arr = []
        if segment == (num_of_segments - 1):
            segment_arr = scaler.fit_transform(data[(segment * scalar_window_size):])
        else:
            segment_arr = scaler.fit_transform(data[(segment * scalar_window_size):((segment + 1) * scalar_window_size)])
        for sample in segment_arr:
            scaled_data.append(sample)

    scaled_data = np.array(scaled_data)

    if is_test_data:
        prediction_scaler.min_, prediction_scaler.scale_ = scaler.min_[0], scaler.scale_[0]


    x_train = []
    y_train = []

    for step in range(num_of_time_steps + look_ahead_num, len(scaled_data)):
        temp_x_train = []
        temp_y_train = []
        for sample_x in range(step - num_of_time_steps - look_ahead_num, step - look_ahead_num):
            temp_x_train.append(scaled_data[sample_x])
        for sample_y in range(step - look_ahead_num, step):
            if is_test_data:
                temp_y_train.append(data[sample_y][0])
            else:
                temp_y_train.append(scaled_data[sample_y][0])
        x_train.append(temp_x_train)
        y_train.append(temp_y_train)

    x_train = np.array(x_train)
    y_train = np.array(y_train)

    return [x_train, y_train]

#creates and returns an untrained gated recurrent unit model based on internally supplied layers
def make_gru_model(train_set):
    train_x, train_y = train_set[0], train_set[1]
    gru_model = Sequential()
    gru_model.add(GRU(100, return_sequences=False, input_shape=(train_x.shape[1], train_x.shape[2])))
    gru_model.add(Dropout(0.3))
    gru_model.add(Dense(train_y.shape[1]))
    gru_model.compile(optimizer='adam', loss='mean_squared_error')
    return gru_model

#creates and returns a long short term memory model based on internally supplied layers
def make_lstm_model(train_set):
    train_x, train_y = train_set[0], train_set[1]
    lstm_model = Sequential()
    lstm_model.add(LSTM(100, return_sequences=False, input_shape=(train_x.shape[1], train_x.shape[2])))
    lstm_model.add(Dropout(0.3))
    lstm_model.add(Dense(train_y.shape[1]))
    lstm_model.compile(optimizer='adam', loss='mean_squared_error')
    return lstm_model

def train_network(train_set, model, num_of_epochs):
    train_x, train_y = train_set[0], train_set[1]
    model.fit(train_x, train_y, batch_size=batch_size, epochs=num_of_epochs, validation_split=0.1)
    return model

#returns the length of the model directory so that new models can be properly named.
def get_latest_model_num():
    model_list = os.listdir("SavedModels")
    return len(model_list)

#Method used to visualise the shape of the closing price of the dataset against date prior to preprocessing .
#This helps to see whether the dataset should be using scalar windows instead of applying a uniform scale to the entirety of the set.
def visualise_data(data):
    plt.figure(figsize=(10, 5))
    plt.title("Close Price History")
    plt.plot(data["Close"])
    #plt.plot(data["Date"], data["Close"])
    plt.xlabel("samples", fontsize=18)
    plt.ylabel("Close Price", fontsize=18)
    plt.show()

#Method takes a predicted and actual numpy array and returns a mean percentage error on all samples in the test set.
def mean_absolute_percentage_error(y_actual, y_predicted):
    summation = 0
    for sample_error in range(len(y_actual)):
        summation += (math.fabs(y_actual[sample_error] - y_predicted[sample_error])/math.fabs(y_actual[sample_error])) * 100
    mape = (1/len(y_actual)) * summation
    return mape

#Method takes predicted and actual values and returns the root-mean-square error metric for the unscaled datasets.
def root_mean_squared_error(y_actual, y_predicted):
    summation = 0
    for sample in range(len(y_actual)):
        summation += math.pow((y_predicted[sample] - y_actual[sample]),2)
    mse = math.sqrt((1/len(y_actual)) * summation)
    return mse

#Residual sum of squares metric required to calculate the coefficient of determination (R2)
def residual_sum_of_squares(y_actual, y_predicted):
    rss = 0
    for sample in range(len(y_actual)):
        rss += math.pow((y_actual[sample] - y_predicted[sample]), 2)
    return rss

#Total sum of squares metric required in order to calculate the coefficient of determination (R2)
def total_sum_of_squares(y_predicted):
    #calculate result mean
    mean = 0
    for y_sample in range(len(y_predicted)):
        mean+= y_predicted[y_sample]
    mean = mean/len(y_predicted)

    tss = 0
    for y_sample in range(len(y_predicted)):
        tss+= math.pow((y_predicted[y_sample] - mean), 2)
    return tss

#Returns the coefficient of determination error metric.
def r2_score(y_actual, y_predicted):
    return 1 - (residual_sum_of_squares(y_actual, y_predicted)/total_sum_of_squares(y_predicted))

def epoch_wise_training(ready_model, train_set, test_set):

    test_set[1] = test_set[1].flatten()
    model_num = get_latest_model_num()


    for epoch in range(1, epoch_num+1):
        ready_model = train_network(train_set, ready_model, 1)
        t_loss = ready_model.history.history["loss"][0]
        v_loss = ready_model.history.history["val_loss"][0]
        predictions = ready_model.predict(test_set[0])
        predictions = prediction_scaler.inverse_transform(predictions)
        predictions = predictions.flatten()

        current_Mape = round(mean_absolute_percentage_error(test_set[1], predictions), 2)
        current_RMSE = round(root_mean_squared_error(test_set[1], predictions), 2)
        current_R2 = round(r2_score(test_set[1], predictions), 3)

        with open("Results/results.csv", 'a') as file:
            writer = csv.writer(file)
            selected_features_string = ""
            for feature in range(len(selected_features)):
                selected_features_string += selected_features[feature] + "-"
            row_values = [("SavedModels/" + str(model_num)), dataset_path, use_window_scalar, epoch, time_step_num, look_ahead_num, net_type, selected_features_string, batch_size, current_RMSE, current_Mape, current_R2, t_loss, v_loss]
            writer.writerow(row_values)

        print("Epoch num: ", epoch)
        print("MAPE: " + str(current_Mape) + "%")
        print("RMSE: " + str(current_RMSE))
        print("R2:", current_R2,"\n")

    ready_model.save("SavedModels/" + str(model_num))

#method to visualise the validation vs training losses to see whether the network is overfitting currently does not work as model.history seems to be wiping itself.
def visualise_validation_vs_training_loss(losses_dataframe):
    plt.figure(figsize=(10, 5))
    plt.title("Loss Comparison")
    plt.plot(losses_dataframe["Validation"])
    plt.plot(losses_dataframe["Training"])
    plt.xlabel("epoch num", fontsize=18)
    plt.ylabel("MSE loss", fontsize=18)
    plt.legend(["Training", "Validation"], loc="lower right")
    plt.show()

#Main method creates models and executes the artifact code
def main():

    user_ui()
    scaler = MinMaxScaler(feature_range=(0,1))
    raw_dataset = read_dataset(dataset_path)
    numpy_feature_filtered_array = filter_dataset(selected_features, raw_dataset)

    #constrains the train data size to 80% of all data
    train_set_size = math.ceil(len(numpy_feature_filtered_array) * 0.8)

    #Certain datasets perform  better if a window scalar is applied
    if use_window_scalar:
        scalar_window_size = int(train_set_size/3)
    else:
        #If window scalar is set to false the window size defaults to train set size.
        scalar_window_size = train_set_size

    #creates x and y train value numpy arrays with 3d shape required by keras layers.
    train_set = create_set(numpy_feature_filtered_array[0:train_set_size], time_step_num, look_ahead_num, scalar_window_size, False, scaler)
    #creates x and y test set keeping the y values unscaled with min max so that exact values can be compared from model prediction.
    #This creates a small issue where metrics such MSE and RMSE are significantly higher than published data as they likely do metric calculations on scaled data.
    test_set = create_set(numpy_feature_filtered_array[(train_set_size-time_step_num -1):], time_step_num, look_ahead_num, (len(numpy_feature_filtered_array) - train_set_size - time_step_num), True, scaler)

    if not load_model:
        #Condtion reads the model selection and creates the specified one.
        if net_type == "GRU":
            ready_model = make_gru_model(train_set)
            epoch_wise_training(ready_model, train_set, test_set)
        else:
            ready_model = make_lstm_model(train_set)
            epoch_wise_training(ready_model, train_set, test_set)
    else:
        #visualise_data(raw_dataset)
        model = keras.models.load_model(path_to_model)
        predictions = model.predict(test_set[0])
        predictions = prediction_scaler.inverse_transform(predictions)
        prepared_prediction = []
        test_vals = []
        predictions = predictions.flatten()
        test_set[1] = test_set[1].flatten()

        for predicted_val in range(0, len(predictions), look_ahead_num ** 2):
            for nums in range(look_ahead_num):
                prepared_prediction.append(predictions[predicted_val + nums])
                test_vals.append(test_set[1][predicted_val + nums])


        results_frame = pd.DataFrame()

        results_frame['Prediction'] = prepared_prediction
        results_frame['Close'] = test_vals
        model.summary()
        display_results_graph(results_frame)

#Functions like a main function in java meaning that this file is only executed if the "main()" exists
if __name__ == '__main__':
    main()







