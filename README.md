# Text Analysis Web Application

This web application allows users to submit text for emotion analysis and receive feedback on their input using a pre-trained machine learning model. Below is a comprehensive guide on the components of the application, including the Django, model training, and HTML template.

## Overview
User Text Input: A text area where users can input their text for analysis.

Text Analysis: The text is sent to the server, processed using a machine learning model, and the predicted emotion is returned.

Result Display: The analysis result is displayed on the same page without requiring a page reload.

## Technologies Used

HTML: For structuring the web page.

CSS: For styling the web page.

JavaScript: For handling form submission and displaying results dynamically.

Django: For server-side processing and handling the text analysis logic.

Machine Learning: For emotion prediction using a logistic regression model.

CSRF Protection: To secure the application against CSRF attacks.
