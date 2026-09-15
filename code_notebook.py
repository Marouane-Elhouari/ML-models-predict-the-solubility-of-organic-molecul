from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
X_train , X_test , y_train , y_test = train_test_split(X , y , test_size = 0.2)