# TA_Zahwa
FEWS Based on IoT dan LSTM.

1. Initial Setup and Imports
This section imports all necessary libraries such as numpy for numerical operations, matplotlib for plotting, pandas for data manipulation, MinMaxScaler for data normalization, and Sequential, LSTM, Dense from Keras for building the neural network.

2. Set Random Seed
This cell sets a random seed for numpy to ensure reproducibility of results in operations that involve randomness, such as weight initialization in neural networks. This helps in getting consistent results across multiple runs.

3. Load Training Data
This cell handles the upload and reading of your primary training data (flood_train.csv). It prompts you to upload the file, then reads it into a pandas DataFrame named df. It intelligently attempts to use a 'timestamp' column as the index. Finally, it prints the head of the DataFrame, the timestamp range, and checks for any missing values to give an initial overview of the data.

4. Load Test Data (Unseen Data)
This section is dedicated to loading a separate dataset (flood_test.csv or flood_2.csv in your case) that the model has not seen during training. This 'unseen' data is crucial for evaluating how well the trained model generalizes to new, real-world data. It follows a similar loading and inspection process as the training data.

5. Select Feature and Initial Plot
Here, the 'depth' column from your main DataFrame (df) is extracted. This is the specific time series variable that the LSTM model will be trained to predict. It's converted to a float32 NumPy array, and an initial plot of this raw data is generated to visually inspect its trends and characteristics.

6. Normalize Data
Data normalization is a critical preprocessing step for neural networks, especially LSTMs. The MinMaxScaler scales the 'depth' values to a range between 0 and 1. This helps in stabilizing the training process, preventing large gradients, and often improves model performance. The data is reshaped to a 2D array as required by the scaler.

7. Split Data into Training and Test Sets
This cell divides the normalized dataset into two parts: a training set (67% of the data) used to train the model, and a test set (33% of the data) used to evaluate the model's performance on data it hasn't seen during training. This split helps assess the model's generalization capabilities on data from the same source.

8. Visualize Training and Test Data
These plots provide a visual separation of the training and test datasets. It allows you to see the portion of the time series that will be used for learning patterns and the portion reserved for evaluating those learned patterns.

9. Create Sliding Window Dataset Function
The create_dataset function is defined here. This function implements the sliding window technique, which is fundamental for time series forecasting with LSTMs. It transforms a univariate series into a supervised learning problem by creating input sequences (dataX) of a specified sliding_window size and corresponding output values (dataY) to be predicted.

10. Apply Sliding Window to Training and Test Data
With the create_dataset function, this cell applies the sliding window transformation to both the train and test datasets. A slide_window of 10 is chosen, meaning the model will use the previous 10 data points to predict the next one. This generates trainX, trainY, testX, and testY.

11. Reshape Data for LSTM Input
LSTM layers in Keras expect input data to be in a 3D format: (samples, timesteps, features). This cell reshapes trainX and testX accordingly. In this case, timesteps is 1 (as each input sequence is treated as a single time step containing multiple features from the sliding window), and features is equal to slide_window (10).

12. Build and Train LSTM Model
This section defines, compiles, and trains the LSTM neural network:

Model Architecture: A Sequential model is created, consisting of an LSTM layer (with 4 units and an input_shape matching our reshaped data) followed by a Dense output layer (with 1 unit to predict the next single value).
Compilation: The model is compiled using mean_squared_error as the loss function (to minimize the difference between predicted and actual values) and the 'adam' optimizer (an efficient algorithm for gradient descent).
Training: The model is trained using model.fit() on trainX and trainY for 70 epochs (passes over the entire training dataset), with a batch_size of 1.
13. Evaluate Model on Training and Test Sets (RMSE)
After training, the model makes predictions on both the training and test input sequences. These predictions are then inverse-transformed (scaled back to the original 'depth' units) to calculate the Root Mean Squared Error (RMSE). RMSE is a common metric to evaluate the accuracy of regression models, indicating the average magnitude of the errors. Lower RMSE values mean better performance.

14. Plot Predictions vs. Original Data
This cell visualizes the model's performance on the training and test datasets. It plots the original 'depth' data, along with the model's predictions for both the training and test periods. This plot helps to visually assess how well the model has learned the underlying patterns and how accurately it predicts future values within the known data range.

15. Detailed Plot of Test Predictions vs. True Test Values
This plot provides a more focused view of the model's performance specifically on the test set. By comparing the inverse-transformed test predictions directly against the true inverse-transformed test values, we can clearly see the model's accuracy on data it has not encountered during training. This helps in understanding the prediction quality and any discrepancies.

16. Load Unseen Data for Prediction
This section handles the loading of an entirely new, unseen dataset (flood_2.csv in this case). This dataset is meant to simulate real-world data that the model has never encountered before. The process involves prompting the user for upload, reading the CSV, and inspecting its initial rows, similar to how the training data was loaded.

17. Extract 'depth' from Unseen Data
Similar to the training data, the 'depth' column is extracted from the unseen DataFrame. This prepares the specific feature that our trained model will attempt to predict in this new dataset.

18. Plot Raw Unseen Data
This cell plots the raw 'depth' values from the unseen dataset. It provides an initial visual understanding of the data's characteristics and trends before any processing or model application.

19. Normalize Unseen Data
The unseen_test data is normalized using the same scaler that was fitted on the original training data. It's crucial to use the same scaling parameters to ensure consistency. The data is also reshaped to the 2D format required by the scaler.

20. Create Sliding Window for Unseen Data
The create_dataset function (which creates the sliding window sequences) is applied to the normalized unseen_clean data. This prepares the input sequences (features) and their corresponding target values (labels) for the model to make predictions on this new data. The features are then reshaped into the 3D format required by the LSTM model.

21. Predict on Unseen Data
With the unseen data prepared in the correct format, the trained LSTM model generates predictions (unseen_results) for the 'depth' values in this new dataset.

22. Plot Predicted vs. Ground Truth for Unseen Data
This cell inverse-transforms both the predicted (unseen_results) and true (labels) values from the unseen dataset back to their original 'depth' scale. It then generates two plots: one for the predicted values and one for the true values. These plots allow for a visual comparison of how closely the model's predictions align with the actual unseen data.

23. Calculate RMSE for Unseen Data
To quantitatively evaluate the model's performance on the unseen data, the RMSE is calculated between the inverse-transformed labels (true values) and unseen_results_inverse (predictions). This RMSE provides a concrete measure of the model's generalization capability to entirely new data points.

24. Detailed Plot of Predicted vs. Ground Truth (First 20000 Points)
This section provides a zoomed-in view of the model's predictions on the unseen data, specifically focusing on the first 20,000 data points. This is useful for observing the model's behavior in more detail, especially in periods where the 'depth' values are not at extreme levels, which, as noted, is where the model is typically more comfortable predicting.

Overall Conclusions
The LSTM model has been successfully implemented, trained, and evaluated for predicting water depth based on historical 'depth' data. Here are the key takeaways:

Data Preprocessing: The 'depth' column was successfully extracted, normalized, and prepared using a sliding window approach, which is crucial for time series forecasting with LSTMs.
Model Training: The LSTM model was trained for 70 epochs, showing a good reduction in loss over time, indicating learning from the training data.
Performance on Original Data:
The Train Score (RMSE) of 6.30 suggests the model fits the training data reasonably well.
The Test Score (RMSE) of 16.56 (on the split portion of the original data) indicates some variance between training and new data from the same source, which is expected.
Generalization to Unseen Data:
The Unseen Test Score (RMSE) of 10.77 is a strong indicator of the model's ability to generalize. It's lower than the initial test score, which could be due to the characteristics of the specific flood_2.csv dataset, or simply better performance on that particular unseen distribution.
The plots of predicted versus ground truth for the unseen data visually confirm that the model is capturing the trends in water depth.
Observation: As noted in the original notebook comments, the model appears to perform relatively well when flood levels are not extreme, which is often a challenge for models if such extreme events are underrepresented in the training data.
Overall, the LSTM model provides a robust framework for predicting water depth, demonstrating satisfactory performance on both internal validation and completely unseen data.
