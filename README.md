### Data and code for "Deep Learning and Satellite Imagery provide highly precise predictions of genetic diversity and differentiation in a subterranean mammal"

The collection of samples for  genetic analyses was carried out in Las Grutas (~ 10 km south of the city of Necochea, province of Buenos Aires, Argentina; 38°37'S; 58°50'W, between March 2003 and April 2005. A total of 112 individuals of the herbivorous subterranean rodent *Ctenomys australis* were live-trapped and a finger snip sample taken and preserved. All individual were released back to their burrows at the point of capture. For each sample individual  the multilocus genotype for 9 microsatellite  loci was obtained. Geographical coordinates and microsatellite data are in file `micros.str`

An rgb image of the coastal landscape including the sampling area (3.5 x 1.5 km, approx.) was downloaded  in jpg format from GoogleEarth. The image corresponds to a pansharpened Quickbird image of  2005-04-17 which includes the area where the genetic samples were collected. We defined a kml rectangle at  Latitude: -38.63466 , -38.62104; Longitude: -58.87851 , -58.83692 and then downloaded the historical jpg image for that date from google earth at the maximum resolution available. We then cropped the jpg image to the inner edge of the polygon line and georeferenced it in R using the `raster`, `sp`, and `gdal` packages to get  rgb bands with a spatial resolution of 0.85 m. 

**Image technical specifications**

DigitalGlobe 2005-04-17

Catalog ID: 10100100042DA105 

Cloud Cover: 0%, Quality: 99

QB02 2005-04-17 0.0\% 7.9°

Image ID: 10100100042DA100

Image Clouds: 0.0%

Image Off Nadir: 9.6°

Bands: 4-BANDS

Max GSD: 0.63m

Sun Elevation: 35.1°

Max Target Azimuth: 353.7°

#### Processing of  genetic and image data

To integrate image and genetic data a script in R handled microstatellite data and geographic coordinates to obtain summary statitics of genetic data and the corresponding image data for one-hectare squares. Two summary statistics were obtained, the mean number of alleles per individual (`mAlleles`) and measure of genetic differentiation of the individuales included in the one-hectare square with respect of the rest of individuals in the sample (`Fst`). Data on the genetic summaries and associated image data were saved to a `.csv` file for predicting genetic data from image data using Deep Learning algorithms.

#### R script for preprocessing data

```
library(raster)
library(dismo)
library(rgdal)
library(hierfstat)
library(pegas)
library(rgeos)
library(data.table)
library(readr)
library(jpeg)


ll="+proj=longlat +datum=WGS84 +no_defs +ellps=WGS84 +towgs84=0,0,0"
utm="+proj=utm +zone=21 +datum=WGS84 +units=m +no_defs +ellps=WGS84 +towgs84=0,0,0"

img1 <- readJPEG("area.jpg")

red1=raster(255*img1[,,1])
green1=raster(255*img1[,,2])
blue1=raster(255*img1[,,3])

area=stack(list(red1, green1, blue1))
proj4string(area)=ll
extent(area)=extent(readOGR("area.kml"))


Rast=projectRaster(area, crs=utm, method="ngb")

australis=read.structure("micros.Stru", n.ind=112, n.loc=9, onerowperind=T, col.lab=1, 
                col.pop=2, col.others = c(3,4), row.marknames=1, NA.char="-9", ask=F)

dfAus=genind2df(australis)
dfAus[,2:10]=as.integer(unlist(dfAus[,2:10]))

tucos = as.data.frame(australis$other)

colnames(tucos)=c("Long", "Lat")
tucos$Long=type.convert(as.character(tucos$Long))
tucos$Lat=type.convert(as.character(tucos$Lat))
coordinates(tucos)=c("Long", "Lat")

proj4string(tucos)=ll
tucos=spTransform(tucos, proj4string(Rast))

extP=extent(c(xmin=336501.5, xmax=340091.5,  ymin=-4277834, ymax=-4276415))

Rast=crop(Rast, y=extP)

D=distanceFromPoints(Rast, tucos)
indi=which(D[]<50)
D[]=NA
D[indi]=1
np=data.frame(randomPoints(D, n=1000))
coordinates(np)=c("x", "y")
proj4string(np)=proj4string(Rast)

for(j in 1:1000){
incluidos=1
  while(length(incluidos)<=6){
    polyG = gBuffer(np[sample(1:1000, size=1)], width=50, capStyle = "SQUARE" )
    incluidos=which(is.na(over(tucos, polyG))==F)
  }

img=crop(Rast, y=polyG)

dfAus$pop=2
dfAus$pop[incluidos]=1

Fst=basic.stats(dfAus)$overall["Fst"]
NumAle=sum(allelic.richness(dfAus[incluidos,])$Ar)

GeneImageData=data.frame(t(c(length(incluidos), extent(polyG)[],  Fst, NumAle, img[])))

write_csv(GeneImageData, "giData.csv", append = TRUE)
}
```
This script saves a `csv` file (`giData.csv`) with the number of individuals sampled in each one-hectare square, the number of alleles in each square, the `Fst` index, and the rgb pixel values for each section of one-hectare squares of the image. The mean number of alleles per individual (`mAlleles`) is obtained by dividing the number of alleles by the number of individuals.


#### Prediciton of genetic indexes from image data

To  explain the spatial variation in genetic diversity and genetic differentiation in *C. australis* using  image data we built a simple convolutional neural network (CNN) that consisted of 3 convolutional layers, 2 pooling layers, a flatten layer, a dense layer and a linear output. A 30% dropout was used to avoid over-fitting (see code). The CNN was coded in Python using keras and tensorflow. The mean squared error was used as loss for fitting the CNN. The data was split in a training set (70%) and a validation set (30%) and the fitting procedure was run for 1000 epochs monitoring the validation loss and recording the model at each improvement of the validation loss. To prevent over-fitting if no improvement was made during 50 epochs the procedure was stopped.

#### Python code for trainning the Convolutional Neural Network
```
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import *
from keras.utils.np_utils import to_categorical
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, LearningRateScheduler, ModelCheckpoint
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.optimizers import Adam

train = pd.read_csv("giData.csv", header=None)

# Fst values to Y_train
Y_train = train[[5]]

# predictors in columns 8 to 41074
X_train = train.drop(train.columns[[range(7)]], axis = 1)
X_train = X_train / 255.0
X_train = X_train.values.reshape(-1,117,117,3, order="F")

# custom R2-score metrics from keras backend
from keras import backend as K

def r2_keras(y_true, y_pred):
    SS_res =  K.sum(K.square(y_true - y_pred)) 
    SS_tot = K.sum(K.square(y_true - K.mean(y_true))) 
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )

# Buil model
model = Sequential()
model.add(Conv2D(64, kernel_size = ks, activation='relu', padding='same', input_shape = (117, 117, 3)))
model.add(MaxPool2D())
model.add(Conv2D(24, kernel_size = 3, activation='relu', padding='same'))
model.add(MaxPool2D(padding='same'))
model.add(Conv2D(48, kernel_size = 3, padding='same', activation='relu'))
model.add(MaxPool2D(padding='same'))
model.add(Conv2D(64, kernel_size = 3, padding='same', activation='relu'))
model.add(Flatten())
model.add(Dropout(0.3))
model.add(Dense(256, activation='relu'))
model.add(Dense(1, activation='linear')))

opt =Adam(lr=0.001)
model.compile(optimizer=opt, loss="mean_squared_error", metrics=[r2_keras])

lr_reducer = LearningRateScheduler(lambda x: 0.001 * 0.995 ** x)
EarLY=EarlyStopping(monitor='val_loss', mode='min', min_delta=0, patience=50, verbose=0, restore_best_weights=True)

filepath = 'modelFst_.h5'
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', mode='min', verbose=1, save_best_only=True)

epochs = 1000

history = model.fit(X_train, Y_train, epochs = epochs,  validation_split = 0.3,  batch_size=100, callbacks=[lr_reducer, EarLY, checkpoint])                           
```
Prediction of genetic indexes from image data using this code yielded r<sup>2</sup> values above 0.99 for `mAlleles` and above 0.98 for `Fst`.

#### Fit with data augmentation
In order to get predictions that would perform better with data beyond those used in training we used data augmentation with `keras` `ImageDataGenerator()` by rotating and vertically and horizontally shifting the original images. This allows the neural network to learn patterns useful in predicting the response under a larger variety of conditions increasing its ability to generalize.

The code was modified to use this generator for the training set and leaving the validation set without augmentation.

```
datagen = ImageDataGenerator(rotation_range=30, width_shift_range=0.3,  height_shift_range=0.3,  validation_split=0.3)

Vdatagen = ImageDataGenerator()

bs=100

aigen=datagen.flow(X_train, Y_train, batch_size=bs)
Vaigen=Vdatagen.flow(X_train, Y_train, batch_size=bs)

tsteps =  X_train.shape[0]*0.7/bs
vsteps =  X_train.shape[0]*0.3/bs

history = model.fit_generator(aigen, epochs = epochs,  validation_data = Vaigen, verbose=1, steps_per_epoch = tsteps, 
                              validation_steps=vsteps, callbacks=[lr_reducer, EarLY, checkpoint])

```
This procedure yielded r<sup>2</sup> values ~ 0.8 for both  `mAlleles` and  `Fst`

#### Random Forest and Landscape Metrics to understand fit to genetic indexes

Because the predictive power of deep learning models is somewhat dampened by the impossibility or difficulty in accurately identifying which variables or traits are the drivers of the great predictive capabilities they show, We estimated the distribution of suitable habitat for tuco-tucos in the area using species distribution models using Maxent and summarizing the distribution of patches of suitable habitat with a set of 65 landscape metrics (that measure variation in shape, area, connectivity, etc.) and selected a subset that was free of multicollinearity to carry out a Random Forest analysis.

#### R code to estimate the distribution of suitable area

The rgb image `Rast` together with geographical coordinates of tuco-tucos `tucos` obtained during the preprocessing of data were used to obtain a raster layer of suitable habitat for tuco-tucos in the study area.

```
D=distanceFromPoints(Rast, tucos)
# bg points at d from presence points >50 m and  <300m
indi=which(D[]>50 & D[]<300)
D[]=NA
D[indi]=1
bg=data.frame(randomPoints(D, n=1000))
coordinates(bg)=c("x", "y")
proj4string(bg)=proj4string(Rast)

modelHabitat = maxent(x=Rast, p=tucos, a=bg)

# threshold for binary layer
e = evaluate(p=tucos, a=bg, model=modelHabitat, x=Rast)
threshold(e)$kappa

pred=predict(modelHabitat, Rast)

Suitable=pred>threshold(e)$kappa
writeRaster(Suitable, "Suitable.tif", format="GTiff")
```

The geotiff file `Suitable.tif` is then used to obtain landscape metrics for patches of suitable habitat for the one-hectare squares previously used for training the Convolutional Neural Network.


#### R code to obtain Landscape Metrics of suitable habitat

```
library(landscapemetrics)

igData=fread("giData.csv", header=F)

x_train <- array_reshape(data.matrix(igData[,-(1:7)])/255, c(nrow(igData),117, 117, 3), order="F")
                        
MetricsLandscape=function(landscape){
  df[1, 1] =lsm_l_ai(landscape)$value
  df[1, 2] =lsm_l_area_cv(landscape)$value
  df[1, 3] =lsm_l_area_mn(landscape)$value
  df[1, 4] =lsm_l_area_sd(landscape)$value
  df[1, 5] =lsm_l_cai_cv(landscape)$value
  df[1, 6] =lsm_l_cai_mn(landscape)$value
  df[1, 7] =lsm_l_cai_sd(landscape)$value
  df[1, 8] =lsm_l_circle_cv(landscape)$value
  df[1, 9] =lsm_l_circle_mn(landscape)$value
  df[1, 10] =lsm_l_circle_sd(landscape)$value
  df[1, 11] =lsm_l_cohesion(landscape)$value
  df[1, 12] =lsm_l_condent(landscape)$value
  df[1, 13] =lsm_l_contag(landscape)$value
  df[1, 14] =lsm_l_contig_cv(landscape)$value
  df[1, 15] =lsm_l_contig_mn(landscape)$value
  df[1, 16] =lsm_l_contig_sd(landscape)$value
  df[1, 17] =lsm_l_core_cv(landscape)$value
  df[1, 18] =lsm_l_core_mn(landscape)$value
  df[1, 19] =lsm_l_core_sd(landscape)$value
  df[1, 20] =lsm_l_dcad(landscape)$value
  df[1, 21] =lsm_l_dcore_cv(landscape)$value
  df[1, 22] =lsm_l_dcore_mn(landscape)$value
  df[1, 23] =lsm_l_dcore_sd(landscape)$value
  df[1, 24] =lsm_l_division(landscape)$value
  df[1, 25] =lsm_l_ed(landscape)$value
  df[1, 26] =lsm_l_enn_cv(landscape)$value
  df[1, 27] =lsm_l_enn_mn(landscape)$value
  df[1, 28] =lsm_l_enn_sd(landscape)$value
  df[1, 29] =lsm_l_ent(landscape)$value
  df[1, 30] =lsm_l_frac_cv(landscape)$value
  df[1, 31] =lsm_l_frac_mn(landscape)$value
  df[1, 32] =lsm_l_frac_sd(landscape)$value
  df[1, 33] =lsm_l_gyrate_cv(landscape)$value
  df[1, 34] =lsm_l_gyrate_mn(landscape)$value
  df[1, 35] =lsm_l_gyrate_sd(landscape)$value
  df[1, 36] =lsm_l_iji(landscape)$value
  df[1, 37] =lsm_l_joinent(landscape)$value
  df[1, 38] =lsm_l_lpi(landscape)$value
  df[1, 39] =lsm_l_lsi(landscape)$value
  df[1, 40] =lsm_l_mesh(landscape)$value
  df[1, 41] =lsm_l_msidi(landscape)$value
  df[1, 42] =lsm_l_msiei(landscape)$value
  df[1, 43] =lsm_l_mutinf(landscape)$value
  df[1, 44] =lsm_l_ndca(landscape)$value
  df[1, 45] =lsm_l_np(landscape)$value
  df[1, 46] =lsm_l_pafrac(landscape)$value
  df[1, 47] =lsm_l_para_cv(landscape)$value
  df[1, 48] =lsm_l_para_mn(landscape)$value
  df[1, 49] =lsm_l_para_sd(landscape)$value
  df[1, 50] =lsm_l_pd(landscape)$value
  df[1, 51] =lsm_l_pladj(landscape)$value
  df[1, 52] =lsm_l_pr(landscape)$value
  df[1, 53] =lsm_l_prd(landscape)$value
  df[1, 54] =lsm_l_rpr(landscape)$value
  df[1, 55] =lsm_l_shape_cv(landscape)$value
  df[1, 56] =lsm_l_shape_mn(landscape)$value
  df[1, 57] =lsm_l_shape_sd(landscape)$value
  df[1, 58] =lsm_l_shdi(landscape)$value
  df[1, 59] =lsm_l_shei(landscape)$value
  df[1, 60] =lsm_l_sidi(landscape)$value
  df[1, 61] =lsm_l_siei(landscape)$value
  df[1, 62] =lsm_l_split(landscape)$value
  df[1, 63] =lsm_l_ta(landscape)$value
  df[1, 64] =lsm_l_tca(landscape)$value
  df[1, 65] =lsm_l_te(landscape)$value
  return(df)
}

x=list()

for(row in 1:nrow(igData){
e=extent(c(unlist(igData[row,2:5])))
landscape=crop(Suitable, e)
proj4string(landscape)=CRS("+proj=utm +zone=21 +datum=WGS84")
extent(landscape)=e
x[[row]]=MetricsLandscape(landscape)
}

LSMetrics=data.frame(matrix(NA, 1000, length(names(x[[1]]))))
colnames(LSMetrics)=names(x[[1]])


for(j in 1:1000){
  LSMetrics[j,]=x[[j]]
}

CNAs=apply(LSMetrics, 2, function(x)sum(is.na(x)==T))
LSMetrics=LSMetrics[,which(CNAs==0)]

Fst=as.vector(unlist(igData[,6]))
MAlleles=as.vector(unlist(igData[,7]/datos[,1]))

LSMetrics=cbind(Fst, MAlleles, LSMetrics)

LSMetrics=na.omit(LSMetrics)

readr::write_csv(LSMetrics, "LSMetrics.csv")                         
```
The file `LSMetrics.csv` contains genetic indexes and 65 landscape metrics for all one-hectare squares. To use these variables in a Random Forest model a subset of 19 variables with pearson correlation below 0.8 was identified with the following R script.

```
library(caret)
x=LSMetrics[,-(1:2)]
# leave out variables without variation
indi=which(apply(x, 2, sd)!=0)
x=x[,indi]
corr_matrix=cor(x)
highCorr=findCorrelation(x=corr_matrix, cutoff=0.8)
cm=cor(x[,-highCorr])
#get the names of uncorrelated variables
colnames(cm)
```
With the subset of uncorrelated variables we trained a Random Forest model to predict the genetic indexes and evaluate how different landscape metrics contribute to the predictions of the model using shapley values. We illustrate the procedure for `mAlleles`.

#### Python code Random Forest model
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
import seaborn as sbn

df = pd.read_csv('LSMetrics.csv')

# Landscape metrics with pairwise corrleation below 0.8

vars = ["lsm_l_area_mn", "lsm_l_area_sd", "lsm_l_cai_cv", "lsm_l_circle_mn", "lsm_l_circle_sd",  "lsm_l_cohesion", "lsm_l_contig_cv", 
         "lsm_l_contig_sd", "lsm_l_dcad" , "lsm_l_dcore_cv",  "lsm_l_enn_cv", "lsm_l_enn_mn", "lsm_l_frac_sd", "lsm_l_gyrate_cv", 
         "lsm_l_lpi" , "lsm_l_mutinf",        "lsm_l_pafrac", "lsm_l_shape_cv", "lsm_l_shape_mn"] 

X =  df[vars]

# The target variable is 'MAlleles'
Y = df['MAlleles']

np.random.seed(0)
# Split the data into train and test data:
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3)

model = RandomForestRegressor(max_depth=100, random_state=0, n_estimators=10000)
model.fit(X_train, Y_train)
```
This provides a  r<sup>2</sup>=0.968 for the validation dataset. To get a hint of how the landscape variables contribute to the prediction of `mAlleles` we use shapley values.

#### Python code for Shapley values

```
import shap
shap_values = shap.TreeExplainer(model).shap_values(X)
vals= np.abs(shap_values).mean(0)
```
Shapley values are now available in `shap_values` and the contribuion of landscape variables are in `vals`. Shapley plots to inspect the contributions can be obtained using  `shap.dependence_plot() `; for example:

```
shap.dependence_plot("lsm_l_shape_cv", shap_values, X)
```


