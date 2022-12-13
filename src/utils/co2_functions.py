import numpy as np
import pandas as pd
import sys
import os
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans,DBSCAN
from sklearn.metrics import silhouette_samples,silhouette_score,confusion_matrix,\
                            classification_report,precision_score,recall_score,\
                            f1_score, r2_score, mean_absolute_error,mean_squared_error
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import cross_val_score,KFold, RepeatedKFold,\
                                    validation_curve, learning_curve,\
                                    RandomizedSearchCV,GridSearchCV
from sklearn.preprocessing import PowerTransformer
import seaborn as sns
from sklearn.feature_selection import RFECV
from typing import Union
sys.path.append(os.getcwd())
import general_purpose as gp
print("Hey!, el módulo co2 ha sido importado correctamente \U0001F973")


class Clustering:

    """Contiene los métodos para la fase de clustering del proyecto co2"""

    def __init__(self):
        pass

    def grafico_silueta(df:pd.DataFrame,estimador:str="k",radio:Union[float,int]=None,
                        minpts:int=None,n_clusters:list=[2,3,4]) -> plt:

        """Realiza el gráfico de silueta para el número de clusters elegidos con KMeans o DBSCAN
            
        --------------------------------------
        # Args:
            - df = pd.DataFrame
            - estimador = (str) k=kmeans o d= DBSCAN
            - radio = (float or int) es el eps de DBSCAN
            - minpts = (int) min_samples para DBSCAN
            - n_clusters = (list) lista con los nº de clusters a mostrar para KMeans
        
        ----------------------------------------
        # Return:
            - gráfico de matplotlib.pyplot"""
        
        if estimador == "k":
            for k in n_clusters:
                fig,ax= plt.subplots(1)
                fig.set_size_inches(25, 7)
                km = KMeans(n_clusters=k)
                labels = km.fit_predict(df)
                centroids = km.cluster_centers_


                silhouette_vals = silhouette_samples(df, labels)

                y_lower, y_upper = 0, 0
                for i, cluster in enumerate(np.unique(labels)):
                    cluster_silhouette_vals = silhouette_vals[labels == cluster]
                    cluster_silhouette_vals.sort()
                    y_upper += len(cluster_silhouette_vals)

                    plt.barh(range(y_lower, y_upper), cluster_silhouette_vals, edgecolor='none', height=1,alpha=0.6)
                    plt.text(-0.03, (y_lower + y_upper) / 2, str(i + 1),weight="bold",fontsize="medium")
                    y_lower += len(cluster_silhouette_vals)

                
                avg_score = np.mean(silhouette_vals)
                plt.axvline(avg_score, linestyle='--', linewidth=5, color='red')
                plt.annotate(f"Silhouette Score:{round(avg_score,3)}",xy=(avg_score*1.1,y_upper/10),
                            fontsize="x-large",fontfamily="fantasy")
                plt.xlim([-0.06, 1])
                plt.xlabel('Silhouette coefficient values')
                plt.ylabel('Cluster labels')
                plt.title('--Silhouette plot for {} clusters--'.format(k), y=1.02,
                            fontsize="x-large",weight="bold",fontfamily="monospace",
                            bbox=dict(boxstyle="Round4",alpha=0.3,color="grey"))
            plt.show()

        if estimador == "d":
            try:
                fig,ax= plt.subplots(1)
                fig.set_size_inches(25, 7)
                km = DBSCAN(eps=radio,min_samples=minpts)
                labels = km.fit_predict(df)
                labels_1 = np.array([x for x in labels if x != -1])
                df2 = df.drop(index=np.where(labels==-1)[0])
                silhouette_vals = silhouette_samples(df2, labels_1)

                y_lower, y_upper = 0, 0
                for i, cluster in enumerate(np.unique(labels_1)):
                    cluster_silhouette_vals = silhouette_vals[labels_1 == cluster]
                    cluster_silhouette_vals.sort()
                    y_upper += len(cluster_silhouette_vals)

                    plt.barh(range(y_lower, y_upper), cluster_silhouette_vals, 
                                    edgecolor='none', height=1,alpha=0.6)
                    plt.text(-0.03, (y_lower + y_upper) / 2, str(i + 1),
                                    weight="bold",fontsize="medium")
                    y_lower += len(cluster_silhouette_vals)

            
                avg_score = np.mean(silhouette_vals)
                plt.axvline(avg_score, linestyle='--', linewidth=5, color='red')
                plt.annotate(f"Silhouette Score:{round(avg_score,3)}",
                            xy=(avg_score*1.1,y_upper/10),
                            fontsize="x-large",fontfamily="fantasy")
                plt.xlim([-0.06, 1])
                plt.xlabel('Silhouette coefficient values')
                plt.ylabel('Cluster labels')
                plt.title('--Silhouette plot for {} clusters--'.format(len(np.unique(km.labels_))-1),
                            y=1.02,fontsize="x-large",weight="bold",fontfamily="monospace",
                            bbox=dict(boxstyle="Round4",alpha=0.3,color="grey"))
                plt.show()
            except:
                print("""
You should change radio and min_samples as it has just made one cluster,beeing 
this -1 (noise), so it is not possible to draw a graph. Probably it would be a
good thing to change the estimator as this is not appropriate for your dataset.
It could also help scaling the values.""")

    def grafico_codo(df:pd.DataFrame,max_clusters:int=9,semilla:int=None,
                    optimo:int=None) -> px:

        """Funcion que grafica el codo en KMeans para detectar los clusters óptimos

        ---------------------------------------
        # Args:
            - df = pd.DataFrame
            - max_clusters = (int) número máximo de clusters ha representar
            - semilla = (int)
            - optimo = (int) se puede poner despues de ver el gráfico para pintar una línea

        ---------------------------------------
        # Return:
            gráfico de plotly.express
        """
        inertia = []
        for i in np.arange(2,max_clusters):
            km = KMeans(n_clusters=i,random_state=semilla).fit(df)
            inertia.append(km.inertia_)

        # ahora dibujamos las difrentes distorsiones o inercias:
        fig = px.line(x= np.arange(2,max_clusters), y= inertia,markers=True,
                        title="|| K-Means Inertia ||",
                        labels=dict(x="clusters",y="inertia")).add_vline(x=optimo,
                                                                line_color="green")
        return fig.show()

    def aplicacion_dbscan(df:pd.DataFrame) -> str:

        """Aplica el algoritmo DBSCAN y nos devuelve el número de clusters,
        los puntos de ruido y el coeficiente de silueta
        ------------------------------------
        # Args:
            - df=(pd.DataFrame)

        ------------------------------------
        # Return:
            (Print) número clusters, puntos de ruido, Silhouette Coefficient
        """
    

        db = DBSCAN().fit(df)
        labels = db.labels_

        # Número de clusters en labels, ignorando el ruido si está presente.
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise_ = list(labels).count(-1)

        print("Número estimado de clusters: %d" % n_clusters_)
        print("Número estimado de puntos como ruido: %d" % n_noise_)
        print("Silhouette Coefficient: %0.3f" % silhouette_score(df, labels))


class Predicting:

    """Contiene los métodos de la fase de Regresión del proyecto co2"""

    def __init__(self):
        pass

    def compute_vif(df:pd.DataFrame,considered_features:list) -> pd.DataFrame:

        """Función que realiza la prueba vif (Variance Inflator Factor ordenado
        de mayor a menor valor
        
        ----------------------------------------------
        # Args:
            - df: (pd.DataFrame) variables a comprobar
            - considered_features: (list) lista de variables a comprobar
        
        -----------------------------------------------
        # Return:
            pd.DataFrame"""
        
        X = df.loc[:,considered_features]
        X['intercept'] = 1
        
        vif = pd.DataFrame()
        vif["Variable"] = X.columns
        vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
        vif = vif[vif['Variable']!='intercept']
        return vif.sort_values(by="VIF",ascending=False).reset_index().drop(columns="index").round(2)


    def cross_val_regression(estimador:any,xtrain:Union[pd.DataFrame,np.ndarray],
                            ytrain:Union[pd.DataFrame,np.ndarray],
                            xtest:Union[pd.DataFrame,np.ndarray],
                            ytest:Union[pd.DataFrame,np.ndarray],pred:str="no",
                            grafico:str="si") -> Union[any,pd.DataFrame,np.ndarray]:
                        
        """Función que devuelve las métricas r2, mae y mse hechas mediante un
        cross validation y puede devolver un gráfico comparativo de la realidad
        con la predicción para comprobar el ajuste del modelo
        
        -----------------------------------------
        # Args:
            - estimador: algoritmo de regresión con métricas r2, mae y mse
            - xtrain: (pd.DataFrame,np.ndarray) features train
            - ytrain: (pd.DataFrame,np.ndarray) target train
            - xtest:  (pd.DataFrame,np.ndarray) features train
            - ytest: (pd.DataFrame,np.ndarray) target test
            - pred: (str) si o no para decidir si devolver las predicciones
            - grafico: (str) si o no para decidir si devuelve el gráfico
        
        -----------------------------------------
        # Return:
            Gráfico de seaborn y pd.DataFrame or np.ndarray"""

        cvr = RepeatedKFold()

        # We first cross validate the data
        r2 = cross_val_score(estimador,xtrain,ytrain,scoring="r2",cv=cvr).mean()
        mae = abs(cross_val_score(estimador,xtrain,ytrain,
                            scoring="neg_mean_absolute_error",cv=cvr).mean())
        mse = abs(cross_val_score(estimador,xtrain,ytrain,
                            scoring="neg_mean_squared_error",cv=cvr).mean())
        median = np.median(abs(cross_val_score(estimador,xtrain,ytrain,
                            scoring="neg_median_absolute_error",cv=cvr)))

        # Then we train and predict with the model:
        lr = estimador
        lr.fit(xtrain,ytrain)
        predic = lr.predict(xtest)

        # r2_test = r2_score(ytest,predic)
        # mae_test = mean_absolute_error(ytest,predic)
        # mse_test = mean_squared_error(ytest,predic)

        val_text = "r2: {}  mae: {}  median_ae: {}  rmse: {}".format(round(r2,3),
                                                        round(mae,3),
                                                        round(median,3),
                                                        round(mse**(1/2),3))

        # test_text = "r2: {}  mae: {}  mse: {}  rmse: {}".format(round(r2_test,3),
        #                                                 round(mae_test,3),
        #                                                 round(mse_test,3),
        #                                                 round(mse_test**(1/2),3))

        # And finally we represent and compare the validation and test data
        box = {"facecolor":"grey", "alpha":0.2}

        if grafico == "si":
            plt.figure(figsize=(10,5))
            sns.scatterplot(x=ytest,y=ytest,label="realidad",color="red")
            sns.scatterplot(x=ytest,y=predic,label="predicción",color="green")
            plt.title("Realidad VS Predicción")

            plt.figtext(s="Validation Metrics",x=0.1287,y=-0.01,va="baseline",
            weight="bold",fontsize="x-large")
            plt.figtext(s=val_text,x=0.1287,y=-0.10,va="baseline",
            bbox=box,weight="bold",fontsize="x-large")

            # plt.figtext(s="Test Metrics",x=0.1287,y=-0.20,va="baseline",
            # weight="bold",fontsize="x-large")
            # plt.figtext(s=test_text,x=0.1287,y=-0.30,va="baseline",
            # bbox=box,weight="bold",fontsize="x-large")

            plt.show()

        if pred == "si":
            return predic

    def sin_multico_unoauno(df:pd.DataFrame,variables:list) -> pd.DataFrame:

        """Función que elimina de una en una las variables con vif superior a 5

        -----------------------------------
        # Args:
            - df: (pd.DataFrame)
            - variables: (list) variables a tener en cuenta
        
        -----------------------------------
        # Return:
            pd.DataFrame
        """
        
        try:
            df_vif = Predicting.compute_vif(df,variables).round(2)
            for vif in range(len(df_vif)):
                if (df_vif.loc[0,"VIF"] >= 5) or (df_vif.loc[0,"VIF"] == np.inf):
                    variables.remove(df_vif.loc[0,"Variable"])
                    df_vif = Predicting.compute_vif(df,variables).round(2)
                    df_vif = df_vif.reset_index().drop(columns="index").round(2)
            return df_vif.round(2)
        except KeyError:
            print("todas las variables han sido eliminadas al estar todas por encima de vif 5")

    def metrics_comparison(x:pd.DataFrame,y:pd.Series,list_est:list) -> pd.DataFrame:

        """Performs a comparison between estimators using mae, rmse and r2 metrics

        -----------------------------------------------
        # Args:
            - x: (pd.DataFrame) features
            - y: (pd.Series) target
            -lista_estimadores: (list) list of estimators to compare

        ----------------------------------------------
        # Returns:
            pd.DataFrame
        """
        resultados = {"r2":[],"mae":[],"median_mae":[]}
        metrics = ["r2","neg_mean_absolute_error","neg_median_absolute_error"]
        cvr = RepeatedKFold()

        for i in range(len(list_est)):
            for j in range(len(metrics)):
                if metrics[j] == "r2":
                    cv = cross_val_score(list_est[i],x,y,scoring=metrics[j],
                                                            cv=cvr).mean()
                else:
                    pre_cv = np.abs(cross_val_score(list_est[i],x,y,scoring=metrics[j],
                                                            cv=cvr))
                    cv = np.median(pre_cv)
                resultados[list(resultados.keys())[j]].append(cv)

        ols_metrics = pd.DataFrame(resultados,
                    index=[str(x)[:40] for x in list_est]).sort_values(by="r2",
                                                                        ascending=False)
        return ols_metrics

    def dataset(df:pd.DataFrame,cl:int,include_vars:list,trans:any) -> pd.DataFrame:
        """Función para crear train, test y cluster dataframes

        ----------------------------------------
        # Args:
            - df: (pd.DataFrame) dataframe original
            - cl: (int) cluster a filtrar
            - include_vars: (list) lista de variables a incluir
            - trans: (sklearn scaler) escalado a aplicar a los datos

        ----------------------------------------    
        # Return:
            pd.Dataframe"""

        df_tuning = df.select_dtypes(exclude="object").drop(["latitude","longitude"],axis=1)
        df_cluster = df_tuning[df_tuning.clusters==cl].reset_index(drop=True)

        exclude_vars = [x for x in df_cluster if x not in include_vars]

        x_train,x_test,y_train,y_test = gp.data_transform(df_cluster,"eficiency",
                                                    trans,
                                                    skip_t=["energy_type"],
                                                    skip_x=exclude_vars,
                                                    test_size=0.3)

        return df_cluster,x_train,x_test,y_train,y_test

    def ml_selector(df_cluster:pd.DataFrame,x_train:pd.DataFrame,
                    x_test:pd.DataFrame,y_train:pd.DataFrame,
                    y_test:pd.DataFrame,estim:list) -> Union[pd.DataFrame,any]:
        
        """Función para validar los modelos, seleccionar el mejor y representarlo
        
        ---------------------------------------------------
        # Args:
            - df_cluster: (pd.DataFrame) dataframe filtrado por cluster
            - x_train: (pd.DataFrame) features para el train
            - x_test: (pd.DataFrame) features para el test
            - y_train: (pd.DataFrame) target para el train
            - y_test: (pd.DataFrame) target para el test
            - estim: (list) lista de estimadores a validar
            
        ---------------------------------------------------
        # Return:
            pd.DataFrame, matplotlib.pyplot"""

        metrics = Predicting.metrics_comparison(x_train,y_train,estim)

        chosen = [x for x in estim if str(x)[:40] == metrics.index[0]][0]

        plot = Predicting.cross_val_regression(chosen,x_train,y_train,x_test,y_test)

        return metrics, plot

    def val_curve_plot(df_cluster:pd.DataFrame,estim:list,include_vars:list,
                                            p_name:str,p_range:any) -> plt:

        """Función que realiza la validation curve en función del parámetro y el
        rango del mismo dado
        
        -----------------------------------------------
        #Args:
            - df_cluster: (pd.DataFrame) datos filtrados por cluster
            - estim: (list) estimador
            - include_vars: (list) features a tener en cuenta
            - p_name: (str) nombre del parámetro a estudiar
            - p_range: (any) rango de valores del parámetro elegido
            
        -----------------------------------------------
        #Return:
            seaborn plot"""

        x = df_cluster[include_vars]
        y = df_cluster.eficiency
        cv =RepeatedKFold()

        train_score,val_score= validation_curve(estimator=estim,X=x,y=y,
            param_name=p_name,param_range=p_range,cv=cv,scoring="r2",)

        sns.lineplot(x=p_range,y=np.mean(train_score,1))
        sns.lineplot(x=p_range,y=np.mean(val_score,1))
        plt.title("Validation Curve")
        plt.ylabel("R2")
        plt.xlabel("Complexity Increase")
        plt.legend(labels=["training_scores","validation_scores"])
        plt.show()
        plt.show()

    def hiper_tune(estimador:any,params:dict,x_train:pd.DataFrame,
                        y_train:pd.DataFrame,scor:str,hiper:str=None) -> any:

        """Función que optimiza los hiperparámetros del estimador dado
        
        -----------------------------------------
        #Args:
            - estimador: (sklearn estimator) estimador para el gridsearchcv
            - params: (dict) parámetros para optimizar
            - x_train: (pd.DataFrame) features
            - y_train: (pd.DataFrame) target
            - scor: (str) scoring con el que seleccionar los mejores parámetros
            - hiper: (str) método de optimización a usar GridSearch(poner grid) 
                        o Randomized (no hace falta poner nada)
            
        -----------------------------------------
        #Return:
            sklearn best estimator (default:RandomizedSearchCV)"""

        cv =RepeatedKFold()
        
        if hiper == "grid":
            grid = GridSearchCV(estimador,param_grid=params,scoring=scor,
            cv=cv).fit(x_train,y_train)
        else:
            grid = RandomizedSearchCV(estimador,param_distributions=params,scoring=scor,
            cv=cv).fit(x_train,y_train)

        print(f"best params: {grid.best_params_}")
        print(f"best score: {grid.best_score_}")

        return grid.best_estimator_

    def learn_curve_plot(df_cluster:pd.DataFrame,include_vars:list,best:any,
                        scor:str) -> any:

        """Función que realiza el gráfico de la learning curve
        
        -------------------------------------------
        #Args:
            - df_cluster: (pd.DataFrame) dataframe del cluster 0
            - include_vars: (list) features a tener en cuenta
            - best: (sklearn estimator) mejor estimador del gridsearchcv
            - scor: (str) métrica para representar
        
        -------------------------------------------
        #Return:
            seaborn plot"""

        x = df_cluster[include_vars]
        y = df_cluster.eficiency

        train_sizes,train_scores,validation_scores = learning_curve(best,
                                                    X=x,y=y,scoring=scor)

        sns.lineplot(x=train_sizes,y=np.mean(train_scores,1))
        sns.lineplot(x=train_sizes,y=np.mean(validation_scores,1))
        plt.title("Learning Curve")
        plt.ylabel(scor)
        plt.xlabel("Training Size")
        plt.legend(labels=["training_scores","validation_scores"])
        plt.show()


class Classification:

    """Contiene los métodos de la fase de Regresión del proyecto co2"""

    def __init__(self):
        pass

    def new_classification_report(realidad:Union[np.ndarray,pd.Series],
                                prediccion:Union[np.ndarray,pd.Series]) -> any:

        """Función que le añade al classification report la confussion matrix

        -----------------------------------
        # Args:
            - realidad: (y_test:np.array | pd.Series) (los datos reales de set de test)
            - prediccion: (pred:np.array | pd.Series) (predicción del estimador)

        ------------------------------------
        # Return:
            Resúmen de las métricas Accuracy,Precision,Recall,F1,support, 
            confussion matrix
        """
        
        sns.heatmap(confusion_matrix(realidad,prediccion),annot=True,
                                    fmt="g",cmap="Greys_r")
        plt.title("Confussion Matrix")
        plt.xlabel("Predicción")
        plt.ylabel("Realidad")
        plt.show()

        print("="*53)
        print(classification_report(realidad,prediccion))
        

    def multiclass_report_bycluster(df:pd.DataFrame,target:str,
                                    l_vars:list,l_estim:list,metrica:str,
                                    splits:int,shuffle:bool=False,
                                    seed:int=None) -> Union[any,pd.DataFrame]:

        """Función que calcula la métrica elegida (Precision,Recall o F1) para cada
        uno de los estimadores seleccionados junto con sus variables elegidas de manera
        individualizada para cada cluster en modelos de clasificación multiclase

        ----------------------------------
        # Args:
            - df: (pd.DataFrame) dataframe completo
            - target: (str) variable objetivo
            - l_vars: (list) lista de listas con las variables para cada estimador
            - l_estim: (list) lista con los estimadores a usar, len(l_estim) = len(l_vars)
            - metrica: (str) Precision, Recall o F1
            - splits: (int) número de splits a realizar por KFold
            - shuffle: (bool) si True aleatoriza las muestras
            - seed: (int) para obtener resultados entre pruebas

        ----------------------------------
        # Return
            pd.DataFrame y plotly.express plot"""

        kf = KFold(n_splits= splits,shuffle=shuffle,random_state=seed)
        df_compar = pd.DataFrame()

        for i,vars in enumerate(l_vars):
            df_x = df[vars]
            s_y = df[target]
            for train_index,test_index in kf.split(df_x):
                x_train,x_test = df_x.iloc[train_index,:], df_x.iloc[test_index,:]
                y_train,y_test = s_y[train_index], s_y[test_index]

                estimator = l_estim[i].fit(x_train,y_train)
                predic = estimator.predict(x_test)
                dic_metrics = {
                "Precision":precision_score(y_test,predic,average=None),
                "Recall": recall_score(y_test,predic,average=None),
                "F1":f1_score(y_test,predic,average=None)
                        }
                metric = dic_metrics[metrica]
                df_compar[str(l_estim[i])] = metric

        df_compar = df_compar.T

        fig = make_subplots(rows=1, cols=len(df_compar.columns),shared_yaxes=True,
                        subplot_titles=["Cluster " + str(x) for x in df_compar.columns])

        for i in range(len(df_compar.columns)):
        
            fig.add_trace(go.Bar(y=df_compar[i],x=df_compar.index,text=round(df_compar[i],3)
                ,marker=dict(color = df_compar[df_compar.columns[0]],colorscale='blugrn',showscale=True)),
                row=1,
                col= i+1)

            fig.update_layout(showlegend=False,template="plotly_dark",
                title_text=f"Mean {metrica} Score for {splits } folds",height=500)

        fig.show()

        return df_compar