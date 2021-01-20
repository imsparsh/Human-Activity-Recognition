# Human Activity Recognition

We will be analyzing the data with respect to all the subjects as we want to classify the activities in general and not be specific to the person.

The modelling for the data will be with respect to the sliding window functionality as the data samples are already sampled with 1 second window with 50% overlap.
This approach will take into consideration the pre-activities that will be involved in the actions being taken by the person as a sequence and will help in making better recurrent neural networks models.

'Working at Computer', 'Standing Up, Walking and Going up\down stairs', 'Standing', 'Walking', 'Going Up\Down Stairs', 'Walking and Talking with Someone', 'Talking while Standing'

## Data Visualization

- line plots for x, y and z for each person to check sample distribution
- box plots for x, y and z for each class to check the outliers
- pie chart for distribution of classes in total
- distplots for x, y and z for gaussian distributions
- countplot for each class per person to check class imbalance

## Data Preparation

- Remove classes with multi-activities that will help in fairer evaluation of the activities which are 0, 2, & 6.
- Train Test Split with first 12 person for training and last 3 person for testing
- Outlier Detection will be applied (values > 3 times SD away from mean should be removed).
- Handle class imbalance problem in data by calculating class weights and setting them while model training.
- Scale the overall data to improve model training.
- Implemented multiple window sizes like 1, 2, 5 seconds to ensure what gives best result.


## Data Modeling

- Modeled with Dense layers of different combination with class weights but couldn't achieve good results.
- Modeled with RandomForest Classifier with class weights but couldn't achieve better results.
- Modeled with different combinations of Conv1D, GRUs, and LSTMs with Dropout and BatchNormalization between the layers with class weights but still could achieve around 64.62% of test accuracy and 65% of recall score.

`Final Result: 3 layer LSTM model with Adam optimizer, with 65% recall rate due to class imabalnce issue which can be seen with recall for specific classes with less samples. These results could be achieved with class weights calculated and can be more improved with generating more augmented data.`

## Future Work

- Oversampling & Undersampling for Class Imabalance issue in the data.
