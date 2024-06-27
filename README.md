**Introduction-**

The goal of this blog is to provide details for how to develop a predictive model for time series forecasting using a recurrent neural network (RNNs). Our specific model will focus on using an RNN to forecast stock prices of Intel. Each data point in a time series dataset is linked to a timestamp, which records the exact time the data was recorded. Fields including finance, economics, weather forecasting, and machine learning often use this type of data. Time series data displays patterns or trends across time. To make predictions on these patterns we can use an RNN.

Recurrent Neural Networks capture the sequential patterns within the data by retaining information about past inputs. As time series data is sequential in nature, it is often used in RNNs. In order to work with time series data and RNNs we can use the TensorFlow library within Python. This blog will go through the steps we took to preprocess our data, engineer our features, select our models, train and evaluate said models, and compare and analyze our models.
Data Preprocessing

In stock price forecasting, we need to prepare our data carefully before feeding it into our neural network. Instead of randomly splitting our data, we divide it based on time. We use historical data from the start of our dataset (date: 2015-01-01) up to the end of 2023 for training. This allows our model to learn from past price movements. For testing, we use data from the beginning of 2024 up to yesterday. This helps us evaluate how well our model predicts future prices.

Before training our model, we need to scale our data. Neural networks, like the Recurrent Neural Networks (RNNs) we're using, work best when all input features are on a similar scale. We use a technique called StandardScaler to scale our data between -1 and 1. This helps prevent issues like vanishing or exploding gradients during training. Essentially, it ensures that no single feature dominates the learning process.

By splitting our data based on time and scaling it appropriately, we set the stage for our RNN model to learn and make accurate predictions about future stock prices.

**Feature Engineering-**

Effective feature engineering can significantly enhance the predictive power of our models. In the context of stock price forecasting, this involves extracting lagged values. These lag values capture historical trends in our data. These lagged values enable our models to learn from past patterns and make informed predictions about future stock prices.I create our lags through a split_sequence function I created.
After I have created the lags, it is possible to split the data into testing and training data. We will train our models on the training data and then evaluate them based on the testing data. Our code applies the split_sequence function to split the scaled training and testing datasets into input-output pairs, with each input sequence containing 3 lagged values. The resulting X_train, y_train, X_test, and y_test contain the input sequences and corresponding output values for training and testing the model.
 

**Model Selection-**

Recurrent neural networks work great for conducting time series forecasting specifically because their ability to train on sequenced data helps to capture temporal dependencies in the data. For this example of predicting stock prices, this is especially important because obviously the days leading up to the day that is being predicted will have some influence or correlation with what the next price will be.
In comparison, other models can fit on this data, but it won’t be performing as well. For instance, I fit a random forest on the data set, still including lagged data to show the time series, and received the following results:
The MSE is: 1.3616922870479322
The R-squared is: 0.9203826551504419
While this may seem good, when predicting on this type of data, there is actually a much higher benchmark. In this scenario, the benchmark of performance is an r-squared value of .93. This is because for predicting time series, specifically for stock prices, the benchmark in performance is to do better than just using the previous day’s price as the prediction, which in this case results in an r-squared of .93. So while other model types than RNNs can fit on this data, and may appear to perform well, the extremely high benchmarks for this problem show that these models simply aren't up to par when it comes to tackling the problem.

**Model Training and Evaluation-**

The RNN model I created for this blog is a Sequential model built using TensorFlow’s Keras API. It is designed to utilize a stack of Simple RNN layers followed by Dense layers for stock price forecasting. The model is first initialized as a Sequential model. RNN layers are added to the model, and each RNN layer is followed by a Dense layer for further processing.
The first Simple RNN layer is added with 60 neurons and an input shape of (3,1). This input shape indicates that the model will receive input sequences with 3 timesteps and 1 feature per timestep. The subsequent Simple RNN layers follow a similar pattern, gradually reducing the number of units from 30 to 20 to 15. Each layer uses the ‘relu’ activation function to introduce non-linearity. We set return_sequences to True for all RNN layers except the last one to ensure each layer returns the full sequence of outputs rather than just the last one.
After the RNN layers, Dense layers were added to the model for further processing and output generation. Two Dense layers were added with 60 and 30 units, both using the ‘relu’ activation. The final Dense layer has 1 unit, which allows for one predicted stock price. After the architecture of the model is created we can compile it using the ‘adam’ optimizer and ‘mse’ as our loss function. I implemented EarlyStopping of 10 steps to prevent the model from running forever. 

Once the model has been fit we can use it to make predictions on our test dataset to evaluate our model.
In order to get the predictions for our model and evaluate it, the first step we need to do is transform the scaled values back to their original values using the “scaler.inverse_transform” after which we can get our predicted values for the X_test. 
We used the MSE and r-squared as our metrics for evaluating the predictions. Our MSE was 1.186 and our r-squared was 0.94 which is just a little bit better than the benchmark performance of 0.93. 
Figure below is the graph of the actual time series and our fitted values.
 

**Model Comparison and Analysis-**

In comparing the performance of the Recurrent Neural Network (RNN) with other traditional machine learning models, such as the Random Forest, using lag 1, we observe notable differences in their predictive capabilities. While the Random Forest model yielded respectable results with an MSE of 1.361 and an R-squared value of 0.920, indicating a strong fit to the data, the RNN model outperformed it with an MSE of 1.186 and an R-squared value of 0.940. Despite the Random Forest's ability to capture some of the temporal dependencies in the data using lagged values, the RNN's inherent capacity to learn sequential patterns and capture temporal dependencies in time series data provides a significant advantage in forecasting stock prices. 

However, it's important to note that each model has its strengths and weaknesses. The Random Forest model excels in interpretability, making it easier to understand the factors driving predictions. Additionally, it requires less computational resources compared to the RNN, making it more computationally efficient, especially for large datasets. On the other hand, the RNN's ability to capture complex temporal patterns and long-term dependencies in the data makes it more accurate and robust for time series forecasting tasks. Despite its computational intensity, the RNN's superior performance justifies its use in scenarios where accuracy is important, such as stock price forecasting. 

The RNN's ability to capture intricate temporal dynamics and provide more accurate predictions makes the computational time it takes worth it, making it the preferred choice for stock price forecasting applications. It is important to note that I only used 3 lags for our predictions, we could add more lags to help the model capture more historical data and its pattern for the predictions, to get better results. 
Conclusion
In conclusion, our exploration into stock price forecasting using (RNNs) has demonstrated the power of deep learning techniques in capturing temporal dependencies and predicting future trends. By leveraging the sequential nature of time series data and engineering informative features, we were able to train an RNN model that outperformed traditional machine learning approaches, such as the Random Forest, in terms of accuracy and predictive capability. While both models offer distinct advantages, the RNN's ability to learn complex patterns and long-term dependencies makes it a superior choice for tasks requiring accurate time series forecasting, such as stock price prediction. Moving forward, the insights gained from this study can inform decision-making processes in finance, economics, and other fields reliant on accurate predictions of sequential data.
