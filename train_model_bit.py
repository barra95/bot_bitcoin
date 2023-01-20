# Codo to train the model
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, plot_confusion_matrix, confusion_matrix, accuracy_score

class Train_Model:
    
    def __init__(self,trans=True):
        self.trans = trans

    def get_train_val(self,dataset): 
        '''Function to split the dataset into Train, Test and Val'''       

        div = round(dataset["signal"].count()*.2,0)

        print(int(div))

        #dataset.iloc[:dataset["signal"].count() - int(div)]
        #dataset.iloc[-int(div):]

        train_dataset = dataset.iloc[:dataset["signal"].count() - int(div)]
        test_dataset= dataset.iloc[-int(div):]

        Y_train= train_dataset["signal"]
        X_train = train_dataset.loc[:, dataset.columns != 'signal']

        Y_validation= test_dataset["signal"]
        X_validation = test_dataset.loc[:, dataset.columns != 'signal']

        return Y_train, X_train, Y_validation, X_validation

    def load_model(self,saved_model_file):
        '''Load Model'''
        saved_model_file = str(saved_model_file )
        file = open(saved_model_file,'rb')
        model = pickle.load(file)
        return model

    def model_training(self,Y_train, X_train, Y_validation, X_validation):
        ''' Function to train the model ''' 
        model = RandomForestClassifier(criterion='gini', n_estimators=80,max_depth=10,n_jobs=-1) # rbf is default kernel
        model.fit(X_train, Y_train) 

        print('Accuracy: ' + str(accuracy_score(Y_validation, model.predict(X_validation))))
       
        if accuracy_score(Y_validation, model.predict(X_validation)) > 0.85:
           model.fit(X_train, Y_train)
        else:
            print('¡¡¡ Revisar Modelo !!!')

        return model

    def print_results(self, model, Y_validation, X_validation):
        ''' Print the results of the model'''
        print('''
        -------------------------------------------------------------------
                                Classification Report
        -------------------------------------------------------------------
        ''')
        print('Accuracy Score: ', round(accuracy_score(Y_validation, model.predict(X_validation)),2))
        print('-------------------------------------------------------------------')
        print(classification_report(Y_validation,model.predict(X_validation)))
        print('-------------------------------------------------------------------')
        print('Confusion Matrix')
        print(confusion_matrix(Y_validation,model.predict(X_validation)))
        plot_confusion_matrix(model, X_validation, Y_validation)

    def saving_model(self,model):

        pkl_filname = 'bitcoin_rf_model.pkl'
        with open(pkl_filname,'wb') as file: 
            pickle.dump(model,file)
