# Algoritmo para predecir divorcios

## Intro

El Dataset es obtenido de [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Divorce+Predictors+data+set) 
En el se encuentran **170 parejas** encuestadas con las siguientes preguntas, en las cuales colocaban una puntuación de 1 a 4 dependiendo la pregunta, como etiqueta se coloca 1 si era divorciado o 0 si no lo era.

1. Si uno de nosotros se disculpa cuando nuestra discusión se deteriora, la discusión termina.
2. Sé que podemos ignorar nuestras diferencias, incluso si las cosas se ponen difíciles a veces.
3. Cuando lo necesitamos, podemos tomar nuestras conversaciones con mi esposo/a desde desde el principio y corregirlo.
4. Cuando discuto con mi esposo/a, contactarlo eventualmente funcionará.
5. El tiempo que paso con mi esposo/a es especial para nosotros.
6. No tenemos tiempo en casa como socios.
7. Somos como dos extraños que comparten el mismo ambiente familiar en la casa.
8. Disfruto las vacaciones con mi esposo/a.
9. Disfruto viajando con mi esposo/a.
10. La mayoría de nuestros objetivos son comunes.
11. Creo que un día en el futuro, cuando mire hacia atrás, veo que mi esposo/a y yo hemos tenido armonía.
12. Mi esposo/a y yo tenemos valores similares en términos de libertad personal.
13. Mi cónyuge y yo tenemos un sentido similar de entretenimiento.
14. La mayoría de nuestras metas para las personas (niños, amigos, etc.) son las mismas.
15. Nuestros sueños con mi esposo/a son similares y armoniosos.
16. Estamos deacuerdo sobre lo que debería ser el amor.
17. Compartimos las mismas opiniones sobre ser feliz en pareja.
18. Mi esposo/a y yo tenemos ideas similares sobre cómo debería ser el matrimonio.
19. Mi esposo/a y yo tenemos ideas similares sobre cómo deben ser los roles en el matrimonio.
20. Mi esposo/a y yo tenemos valores similares en la confianza.
21. Sé exactamente lo que le gusta a mi esposo/a.
22. Sé cómo mi esposo/a quiere ser cuidado cuando esta enfermo.
23. Conozco la comida favorita de mi esposo/a.
24. Puedo decirte qué tipo de estrés enfrenta mi esposo/a en su vida.
25. Tengo conocimiento del mundo interior de mi esposo/a.
26. Conozco las ansiedades básicas de mi esposo/a.
27. Sé cuáles son las fuentes actuales de estrés de mi esposo/a.
28. Conozco las esperanzas y deseos de mi esposo/a.
29. Conozco muy bien a mi esposo/a.
30. Conozco a los amigos de mi esposo/a y sus relaciones sociales.
31. Me siento agresivo cuando discuto con mi esposo/a.
32. Cuando discuto con mi esposo/a, generalmente uso expresiones como "siempre" o "nunca".
33. Puedo usar declaraciones negativas sobre la personalidad de mi esposo/a durante nuestras discusiones.
34. Puedo usar expresiones ofensivas durante nuestras esposo/a.
35. Puedo insultar a mi esposo/a durante nuestras discusiones.
36. Puedo ser humillante cuando discutimos.
37. Las discusiones con mi esposo/a no es tranquila.
38. Odio la forma en que mi esposo/a abre un tema.
39. Nuestras discusiones a menudo ocurren repentinamente.
40. Comenzamos una discusión antes de saber qué está pasando.
41. Cuando hablo con mi esposo/a sobre algo, mi calma de repente se rompe.
42. Cuando discuto con mi esposo/a, solo salgo y no digo una palabra.
43. Principalmente me quedo en silencio para calmar un poco el ambiente.
44. A veces pienso que es bueno para mí salir de casa por un tiempo.
45. Prefiero guardar silencio que discutir con mi esposo/a.
46. Incluso si estoy en lo cierto en la discusión, me quedo callado para lastimar a mi esposo/a.
47. Cuando hablo con mi esposo/a, me quedo callado porque tengo miedo de no poder controlar mi ira.
48. Me siento bien en nuestras discusiones.
49. No tengo nada que ver con lo que me acusan.
50. En realidad no soy el culpable de lo que me acusan.
51. No soy yo quien se equivoca acerca de los problemas en el hogar.
52. No dudaría en contarle a mi esposo/a sobre su insuficiencia.
53. Cuando discuto, le recuerdo a mi esposo/a su insuficiencia.
54. No tengo miedo de contarle a mi esposo/a sobre su incompetencia.

### Modelos Utilizados

Se utiliza las siguientes librería:
- Pandas
- Numpy 
- Seaborn
- Sklearn (Modelos):
  - RandomForest
  - KNeighborsClassifier
  - SVC
- Sklearn (Métricas):
 - Accuracy_score
 - confusion_matrix
 - roc_auc_score
 - recall_score

## Modelo

Los datos estan muy bien balanceados, son 84 divorciados y 86 en matrimonio por lo uqe podemos preceder a alimentar el modelo, antes de eso sume todos los valores de las variables para ver si se encuentra un patrón, el cual si hay, hay preguntas donde solo los divorciados respondian con 0 y otros donde la mayoría de las personas ponian 0.

El primer modelo que utilizamos es **KNeighborsClassifier**, el cual logra muy buenos resultados 97.05 en acuracy y 97.03 en roc_score por lo que hago una matriz de confusión para ver.

![Confusión](https://github.com/rogerzadi/Fraud_Detection_credit_card/blob/master/images/impgra.png)

Como podemos ver solo tuvo un falso negativo, de igual manera gráfico roc_curve y su probabilidad.

![ROU](https://github.com/rogerzadi/Fraud_Detection_credit_card/blob/master/images/impgra.png)

![ROU PROB](https://github.com/rogerzadi/Fraud_Detection_credit_card/blob/master/images/impgra.png)

Ahora me dispongo a ver cuales son las variables de más importancia, por lo que inicialmente hago una matriz de correlación

![CORR](https://github.com/rogerzadi/Fraud_Detection_credit_card/blob/master/images/impgra.png)

En la cual sale:

1. Pregunta 40 con .938684
2. Pregunta 17 con .929346
3. Pregunta 19 con .928627
4. Pregunta 18 con .923208
5. Pregunta 11 con .918386

De igual manera entrene un modelo de randomForest ya que podemos ver cuales son las variables más importantes en su entrenamiento, cabe destacar que al la muestra (80% de los datos) ser aleatoria puede cambiar algo el orden de imporatncia con la lista anterior.

1. Pregunta 9 con 0.371991
2. Pregunta 17 con 0.183089
3. Pregunta 26 con 0.105336
4. Pregunta 19 con 0.104584
5. Pregunta 11 con 0.102210
6. Pregunta 18 con 0.101134

(Este segundo, en total suma 1) 




