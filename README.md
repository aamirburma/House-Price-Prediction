# House Price Prediction

## **Project Overview**
The goal of this project is to predict the sales price (`SalePrice`) of residential homes based on various explanatory variables. This is a regression task using machine learning techniques like **Random Forest** and **Gradient Boosting**. The dataset comes from a competition on predicting house prices in Ames, Iowa, and includes 79 features related to the houses' characteristics.

### **Dataset Description**
- **train.csv**: Contains training data with features and target values (SalePrice).
- **test.csv**: Contains test data with features only (used to make predictions).
- **data_description.txt**: Detailed explanation of each feature.
- **sample_submission.txt**: Format for the submission file, showcasing a benchmark linear regression model.

## **Objective**
- Predict the `SalePrice` for homes in the test set based on the features provided in the training set.
- Evaluate models using **Root Mean Squared Error (RMSE)**, which is calculated on the logarithmic scale of the sales price (`log(SalePrice)`).
- Submit the predicted values in a specific CSV format.

## **Approach**

### 1. **Data Preprocessing**
   - **Missing Value Handling**: Missing values were imputed using the median for numerical features and the mode for categorical features.
   - **Categorical Encoding**: Categorical features were transformed using One-Hot Encoding to create binary variables for each category.
   - **Feature Scaling**: While scaling wasn't strictly necessary for the models chosen, it was considered for potential improvements.
   - **Log Transformation**: To meet the competition's requirements, `SalePrice` was log-transformed using `np.log1p()` to reduce skewness and improve model performance.

### 2. **Modeling**
   - **Random Forest Regressor** and **Gradient Boosting Regressor** were used for prediction. Both models were chosen based on their ability to handle a large number of features and non-linear relationships between variables.
   - **Hyperparameter Tuning**: While basic hyperparameters were set for both models, further tuning using techniques like Grid Search could improve performance.
   - **Validation**: The models were evaluated using RMSE on a validation set created by splitting the training data.

### 3. **Evaluation**
   - RMSE was used to evaluate the performance of both models. The **Random Forest Regressor** model performed better in terms of RMSE and was chosen as the final model for prediction on the test set.

### 4. **Prediction**
   - Predictions were made on the test set using the trained model.
   - Log transformation was reversed (`np.expm1()`) on the predictions to return them to the original scale.

### 5. **Submission**
   - The final predictions were saved in a CSV file with the format:
     ```
     Id, SalePrice
     1461, 169000.1
     ```

---

## **Getting Started**

### **Dependencies**
Ensure you have the following Python libraries installed:
- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scikit-learn`

You can install them using pip:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

### **How to Run the Code**
1. Clone the repository to your local machine.
   ```bash
   git clone https://github.com/aamirburma/house-price-prediction.git
   ```

2. Navigate to the project directory and open the Jupyter Notebook.
   ```bash
   cd house-price-prediction
   jupyter notebook
   ```

3. Run the cells in the notebook to load, preprocess the data, build the models, and generate predictions.

4. After running the notebook, the final predictions will be saved in a `submission.csv` file, which can be submitted to the competition platform.

### **Files in the Repository**
- **`house-price-prediction.ipynb`**: Jupyter notebook containing the entire analysis, from data loading and preprocessing to model evaluation and submission.
- **`train.csv`**: Training dataset with features and target values.
- **`test.csv`**: Test dataset with features only (used for predictions).
- **`data_description.txt`**: Descriptions of the data columns.
- **`sample_submission.txt`**: Sample submission file format.
- **`submission.csv`**: Final submission file generated after running the model.

---

## **Results**

### **Model Evaluation:**
- **Random Forest Regressor RMSE**: `x.xxx`
- **Gradient Boosting Regressor RMSE**: `x.xxx`

In this project, **Random Forest** was found to be the best-performing model based on RMSE. The model was used to predict house prices in the test set.

---

## **Future Improvements**
- **Hyperparameter Optimization**: Use Grid Search or Randomized Search to tune model hyperparameters.
- **Cross-Validation**: Implement K-fold cross-validation for more robust evaluation.
- **Feature Engineering**: Explore additional features or combinations of features to improve model performance.

---

## **Conclusion**
This project demonstrates how machine learning techniques like Random Forest and Gradient Boosting can be applied to predict house prices. By leveraging data preprocessing, feature engineering, and model evaluation techniques, we were able to generate reliable predictions.

Feel free to review the notebook and experiment with different models or enhancements to improve the predictions further!

---

### **License**
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
