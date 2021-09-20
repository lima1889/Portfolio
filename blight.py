'''
Esse projeto usa uma base de dados da cidade de Detroit sobre infrações.
O objetivo é conseguir determinar com o máximo de precisão se uma infração será paga no prazo ou não através dos parâmetros existentes.
Os dois arquivos que acompanham esse script são de treino e teste para o modelo. O modelo train possui parâmetros que o test não possui, como o compliance que representa o resultado do ticket ser pago ou não.
Eu utilizei para esse problema um modelo de RandomForestClassifier por ser eficiente em avaliar diversos parâmetros e prevenir overfitting.
'''

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

def blight_model():
    train_import = pd.read_csv('train.csv', encoding='unicode_escape', low_memory=False)
    test_set = pd.read_csv('test.csv', encoding='unicode_escape', low_memory=False)

    train_set = train_import[train_import.compliance.notnull()]
    exclude_data_train = ['compliance_detail', 'compliance', 'payment_date', 'payment_status', 'payment_amount',
                          'balance_due', 'collection_status', 'violator_name', 'violation_zip_code', 'clean_up_cost',
                          'grafitti_status', 'mailing_address_str_name', 'mailing_address_str_number', 'city', 'state',
                          'zip_code', 'non_us_str_code', 'hearing_date']
    X_pre_train, y_pre_train = train_set.drop(exclude_data_train, axis=1), train_set.iloc[:, -1:]
    X_pre_train = X_pre_train.apply(LabelEncoder().fit_transform)
    y_pre_train.values.reshape(-1, 1)

    X_train, X_t_test, y_train, y_t_test = train_test_split(X_pre_train, y_pre_train)

    rfc = RandomForestClassifier(n_estimators=100, oob_score=True)
    rfc.fit(X_train, y_train['compliance'])

    exclude_data_test = ['violator_name', 'violation_zip_code', 'clean_up_cost', 'grafitti_status',
                         'mailing_address_str_name', 'mailing_address_str_number', 'city', 'state', 'zip_code',
                         'non_us_str_code', 'hearing_date']
    X_pre_test = test_set.drop(exclude_data_test, axis=1)
    X_test = X_pre_test.apply(LabelEncoder().fit_transform)

    '''
    ans = pd.Series(data=rfc.predict_proba(X_test)[:, 1], index=test_set['ticket_id'])
    ans retorna as probabilidades estimadas e foi usado pelo avaliador do curso para computar a AUC.
    Como eu não tinha acesso ao y_true não consigo estimar o score/AUC.
    A área da curva ROC foi reportada como 0.744357884037 pelo avaliador, gerando uma pontuaćão de 96/100 já que a meta era 0.75
    '''

    score = rfc.score(X_t_test, y_t_test)
    '''Esse é o score do teste gerado a partir do split'''

    return score

print(blight_model())
