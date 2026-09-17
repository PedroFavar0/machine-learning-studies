from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import pandas as pd
path_file = 'student-mat.csv'

dataframe = pd.read_csv(path_file, sep=';')
# print(dataframe.columns.tolist())

dataframe.columns = [
    'Escola', 'Sexo', 'Idade', 'Endereço', 'Tamanho familia', 'Situação dos pais', 
    'Educação mãe', 'Educação pai', 'Trabalho pai', 'Trabalho mãe', 'motivo escolha escola',
    'Responsável legal', 'Tempo deslocamento', 'Tempo de estudo semanal', 'Reprovações', 'Suporte extra escolar',
    'Suporte educacional da familia',  'Pagas', 'Atividades extracurriculares' ,'Frequentou pré escola?',
    'Pretende ensino superior?', "Acesso a internet", 'Relacionamento romantico', 'Qualidade rel familia', 
    'Tempo livre', 'Sai com amigos', 'Alcool dia util', 'Alcool fim de semana', 'Saude', 'Faltas', 'G1', 'G2', 'G3'
]

# print(dataframe.columns.tolist())

df_familia = dataframe[['Educação mãe', 'Educação pai', 'Trabalho mãe', 'Trabalho pai', 'Responsável legal',
                       'Qualidade rel familia', 'Suporte educacional da familia', 'Tamanho familia', 
                       'Situação dos pais']]

df_info_aluno = dataframe[['Escola','Sexo', 'Idade', 'Endereço', 'G1', 'G2', 'G3']]

# Médias por escolas
media_GP = df_info_aluno.loc[df_info_aluno['Escola'] == 'GP', 'G3'].mean()
media_MS = df_info_aluno.loc[df_info_aluno['Escola'] == 'MS', 'G3'].mean()

# print(df_info_aluno.head(10))
# print(f"Média alunos GP: {media_GP:.2f} \n Média alunos MS: {media_MS:.2f}")

features = ['Idade', 'Educação mãe', 'Educação pai', 'Tempo deslocamento', 'Tempo de estudo semanal',
            'Reprovações', 'Qualidade rel familia', 'Tempo livre', 'Sai com amigos',
            'Alcool dia util', 'Alcool fim de semana', 'Saude', 'Faltas', 'G1', 'G2']

X = dataframe[features]
# print(X.columns.tolist())
y = dataframe['G3']


train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

modelo_notas = RandomForestRegressor(random_state=1)

modelo_notas.fit(train_X, train_y)

notas_predict = modelo_notas.predict(val_X)

MAE_predict_notas = mean_absolute_error(val_y, notas_predict)

'''print(f"Mean Absolute Error Modelo Notas Try 1: {MAE_predict_notas}") 
-- 1.055959595959596 - sem determinar numero máximo de arvores.'''

def pegar_MAE_estimadores(numero_estimadores, train_X, val_X, train_y, val_y):
    modelo = RandomForestRegressor(
        n_estimators=numero_estimadores,
        random_state=1
    )
    modelo.fit(train_X, train_y)
    predict = modelo.predict(val_X)
    mae = mean_absolute_error(val_y, predict)

    return mae

def pegar_MAE_depth(profundidade, train_X, val_X, train_y, val_y):
    modelo = RandomForestRegressor(
        max_depth=profundidade,
        random_state=1
    )
    modelo.fit(train_X, train_y)
    predict = modelo.predict(val_X)
    mae = mean_absolute_error(val_y, predict)


    return mae

def pegar_MAE_depth_estim(profundidade, numero_estimadores, train_X, val_X, train_y, val_y):
    modelo = RandomForestRegressor(
        n_estimators=numero_estimadores,
        max_depth=profundidade,
        random_state=1
    )
    modelo.fit(train_X, train_y)
    predict = modelo.predict(val_X)
    mae = mean_absolute_error(val_y, predict)


    return mae


candidatos_estimators = [10, 25, 50, 100, 500, 1000, 2000, 2500, 3000]

'''Teste de calibração dos MAE no hiperparâmetro n_estimators. Resultado final ===> 2500 ideal'''

# for n in candidatos_estimators:
#     mae = pegar_MAE_estimadores(n, train_X, val_X, train_y, val_y)
#     print(f'Com {n} estimators temos o MAE: {mae}')

candidatos_profundidade = [1, 3, 5,6, 7, 10]

'''Teste de calibração dos MAE no hiperparametro max_depth. Resultado ideal ===> 5'''

for n in candidatos_profundidade:
    mae = pegar_MAE_depth(n, train_X, val_X, train_y, val_y)
    print(f'Com profundidade {n} temos o MAE: {mae}')


for n in candidatos_estimators:
    for i in candidatos_profundidade:
        pegar_MAE_depth_estim(n, i, train_X, val_X, train_y, val_y )
        print(f'MAE com {n} árvores e {i} perguntas: {mae}')






