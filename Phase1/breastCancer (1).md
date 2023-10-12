# 1. Problem
   The problem is the accurate diagnosis of breast masses as malignant (cancerous) or benign (non- cancerous) using computed features from digitized images of fine needle aspirate (FNA). The features consist of radius, texture,  perimeter,  area, smoothness, compactness,  concavity, concave.points, symmetry, and fractal_dimension of the mass,and there mean, worst, and standard error(SE).
    
   The goal is to predict the nature of breast tumors based on the analyzed features, enabling medical professionals to make informed decisions regarding patient treatment and care.

   Solving the problem of diagnosing breast cancer is important because it can lead to early detection, appropriate treatment plans, improved survival rates, and optimal allocation of healthcare resources.

# 2. Data Mining Task
   The problem can be defined as a data mining binary classification task. The task involves training models and algorithms on the features of breast masses to predict whether a specific mass is malignant(M) or benign(B). The objective is to optimize the classification model's accuracy and performance in differentiating between the two classes using the dataset as the training data. Data mining techniques such as feature selection, model training, and performance evaluation would be employed to solve this task efficiently.

# 3. Data
   We will go over a detailed description of the dataset used in this project. The dataset in focus is the Breast Cancer Wisconsin (Diagnostic) dataset, which plays a crucial role in achieving the goal of accurate breast cancer diagnosis and classification.
   
 ### The goal of collecting this dataset 
the goal of the Breast Cancer Wisconsin (Diagnostic) dataset is to aid in the diagnosis and classification of breast masses as either malignant (cancerous) or benign (non-cancerous).

By analyzing the computed features of cell nuclei extracted from digitized images of fine needle aspirate (FNA), the dataset aims to provide a foundation for developing models and algorithms that can accurately predict the nature of breast tumors.

The ultimate objective is to enhance the accuracy and efficiency of diagnosing breast cancer, enabling medical professionals to make informed decisions regarding patient treatment and care.
   
### The source of the dataset

https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data

### General information about the dataset 

1- Number of attributes (variables, or columns): 32  (the last column (X) Contains only missing "NA" values, that why we didn't count it, and it well be removed later on)

2- Number of objects (records, or rows): The dataset contains a total of 569 instances, each representing a sample of a breast mass.

3- all variables are numeric, except id, and diagnosis.

4- The ID number is a nominal attribute

5- diagnosis is a binary (categoric, or factor) attribute and has two levels of values "B","M".

6- Class name or labels: The diagnosis attribute represents the class labels, with "M" indicating malignant (cancerous) tumors and "B" indicating benign (non-cancerous) tumors.

7- Class distribution: The dataset is imbalanced, with 357 instances classified as benign and 212 instances classified as malignant.

### Attributes table
   The attributes table present important information about the attributes associated with the dataset. This table serves as a reference guide, outlining the different attributes, their description, data type, and possible values. It aims to provide a comprehensive overview of the dataset, enabling us to understand and utilize the information effectively.
<table>
<tr>
    <th>No</th> 
    <th>Attributes</th>
    <th>Description</th>
    <th>Data type</th> 
    <th>Possible values</th> 
</tr>

<tr>
    <th>1</th> 
    <th>id</th>
    <th>Id number for applicant</th>
    <th>nominal</th> 
    <th>Range between 8670-911320502</th> 
</tr>
  
<tr>
    <th>2</th> 
    <th>diagnosis</th>
    <th>indicate whether a case is classified as malignant (cancerous) or benign. 
</th>
    <th>(Asymmetric Binary)categorical</th> 
    <th>"M" for malignant
    "B" for benign</th> 
</tr>

<tr>
    <th>3</th> 
    <th>radius_mean</th>
    <th>Radius: is a measurement of the average distance from the center of the nucleus to its boundary. Mean: mean of distances from center to points on the perimeter. </th>
    <th>numerical</th> 
    <th>Range between 6.9-28.1</th> 
</tr>  
    
<tr>
    <th>4</th> 
    <th>texture_mean</th>
    <th>standard deviation of gray-scale values.</th>
    <th>numerical</th> 
    <th>Range between 9.7-39.2</th> 
</tr>

<tr>
    <th>5</th> 
    <th>perimeter_mean</th>
    <th>Perimeter: total length of the boundary of the nucleus. Mean: mean size of the core tumor</th>
    <th>numerical</th> 
    <th>Range between 43.7-188.5</th> 
</tr>
  
<tr>
    <th>6</th> 
    <th>area_mean</th>
    <th>The mean value of total area occupied by the nucleus. 
</th>
    <th>numerical</th> 
    <th>Range between 143.5-782.7</th> 
</tr>

<tr>
    <th>7</th> 
    <th>smoothness_mean</th>
    <th>local variation in the radius lengths of the cell nuclei present in a breast mass. It quantifies the smoothness of the boundaries of the nuclei. 
</th>
    <th>numerical</th> 
    <th>Range between 0.05-o.16</th> 
</tr> 
    
<tr>
    <th>8</th> 
    <th>compactness_mean</th>
    <th>measures the smoothness of the boundaries and relates to the compactness of the shape of the nuclei. mean of perimeter^2/ area - 1.0 
</th>
    <th>numerical</th> 
    <th>Range between 0.01-0.34</th> 
</tr>
  
<tr>
    <th>9</th> 
    <th>concavity_mean</th>
    <th>mean of severity of concave portions of the contour 
</th>
    <th>numerical</th> 
    <th>Range between 0.0-0.4</th> 
</tr>

<tr>
    <th>10</th> 
    <th>concave.points_mean</th>
    <th>mean for number of concave portions of the contour 
</th>
    <th>numerical</th> 
    <th>Range between 0.0-0.2</th> 
</tr>  
    
<tr>
    <th>11</th> 
    <th>symmetry_mean</th>
    <th>Mean quantifies the degree to which the shape of the nuclei is symmetrical. 
</th>
    <th>numerical</th> 
    <th>Range between 0.1-0.3</th> 
</tr>

<tr>
    <th>12</th> 
    <th>fractal_dimension_mean</th>
    <th>quantifies the complexity and irregularity of the nuclei's shape using fractal geometry. mean for "coastline approximation" - 1 
</th>
    <th>numerical</th> 
    <th>Range between 0.04-0.09</th> 
</tr>
  
<tr>
    <th>13</th> 
    <th>radius_se</th>
    <th>standard error for the mean of distances from center to points on the perimeter 
</th>
    <th>numerical</th> 
    <th>Range between 0.1-2.8</th> 
</tr>

<tr>
    <th>14</th> 
    <th>texture_se</th>
    <th>standard error for standard deviation of gray-scale values</th>
    <th>numerical</th> 
    <th>Range between 0.3-4.8</th> 
</tr> 
    
<tr>
    <th>15</th> 
    <th>perimeter_se</th>
    <th>Standard error for mean size of the core tumor</th>
    <th>numerical</th> 
    <th>Range between 0.7-21.9</th> 
</tr>
  
<tr>
    <th>16</th> 
    <th>area_se</th>
    <th>Standard error for The mean value of total area occupied by the nucleus. 
</th>
    <th>numerical</th> 
    <th>Range between 6.8-542.2</th> 
</tr>

<tr>
    <th>17</th> 
    <th>smoothness_se</th>
    <th>standard error for local variation in radius lengths 
</th>
    <th>numerical</th> 
    <th>Range between 0.001-0.0311</th> 
</tr>  
    
<tr>
    <th>18</th> 
    <th>compactness_se</th>
    <th>standard error for perimeter^2 / area -1.0 
</th>
    <th>numerical</th> 
    <th>Range between 0.002-0.135</th> 
</tr>

<tr>
    <th>19</th> 
    <th>concavity_se</th>
    <th>standard error for severity of concave portions of the contour</th>
    <th>numerical</th> 
    <th>Range between 0.00-0.39</th> 
</tr>
  
<tr>
    <th>20</th> 
    <th>concave.points_se</th>
    <th>standard error for number of concave portions of the contour</th>
    <th>numerical</th> 
    <th>Range between 0.00-0.05</th> 
</tr>

<tr>
    <th>21</th> 
    <th>symmetry_se</th>
    <th>Standard error for the mean that quantifies the degree to which the shape of the nuclei is symmetrical. </th>
    <th>numerical</th> 
    <th>Range between 0.007-0.078</th> 
</tr> 
    
<tr>
    <th>22</th> 
    <th>fractal_dimension_se</th>
    <th>standard error for "coastline approximation" - 1 </th>
    <th>numerical</th> 
    <th>Range between 0.000-0.029</th> 
</tr>
  
<tr>
    <th>23</th> 
    <th>radius_worst</th>
    <th>"worst" or largest mean value for mean of distances from center to points on the perimeter </th>
    <th>numerical</th> 
    <th>Range between 7.93-36.04</th> 
</tr>

<tr>
    <th>24</th> 
    <th>texture_worst</th>
    <th>"worst" or largest mean value for standard deviation of gray-scale values 
</th>
    <th>numerical</th> 
    <th>Range between 12.02-49.54</th> 
</tr>  
    
<tr>
    <th>25</th> 
    <th>perimeter_worst</th>
    <th>"worst" or largest mean value for the size of the core tumor </th>
    <th>numerical</th> 
    <th>Range between 50.41-251.20</th> 
</tr>

<tr>
    <th>26</th> 
    <th>area_worst</th>
    <th>"worst" or largest mean value for total area occupied by the nucleus. </th>
    <th>numerical</th> 
    <th>Range between 185.2-4254.0</th> 
</tr>
  
<tr>
    <th>27</th> 
    <th>smoothness_worst</th>
    <th>"worst" or largest mean value for local variation in radius lengths</th>
    <th>numerical</th> 
    <th>Range between 0.07-0.22</th> 
</tr>

<tr>
    <th>28</th> 
    <th>compactness_worst</th>
    <th>"worst" or largest mean value for perimeter^2 / area - 1.0 
</th>
    <th>numerical</th> 
    <th>Range between 0.02-1.05</th> 
</tr> 
    
<tr>
    <th>29</th> 
    <th>concavity_worst</th>
    <th>"worst" or largest mean value for severity of concave portions of the contour 
</th>
    <th>numerical</th> 
    <th>Range between 0.00-1.2</th> 
</tr>

<tr>
    <th>30</th> 
    <th>concave.points_worst</th>
    <th>"worst" or largest mean value for number of concave portions of the contour 
</th>
    <th>numerical</th> 
    <th>Range between 0.00-0.29</th> 
</tr>  
    
<tr>
    <th>31</th> 
    <th>symmetry_worst</th>
    <th>"worst" or largest mean value that quantifies the degree to which the shape of the nuclei is symmetrical. 
</th>
    <th>numerical</th> 
    <th>Range between 0.1-0.6</th> 
</tr>

<tr>
    <th>32</th> 
    <th>fractal_dimension_worst</th>
    <th>"worst" or largest mean value for "coastline approximation" - 1 </th>
    <th>numerical</th> 
    <th>Range between 0.05-0.20</th> 
</tr>
  
<tr>
    <th>33</th> 
    <th>X</th>
    <th>has no value all records are missing "NA" </th>
    <th>logical </th> 
    <th>"NA"</th> 
</tr>  
</table>

# 4.Code

import and absorve the dataset


```R
data <- read.csv("data.csv")
```


```R
head(data)
```


<table>
<thead><tr><th scope=col>id</th><th scope=col>diagnosis</th><th scope=col>radius_mean</th><th scope=col>texture_mean</th><th scope=col>perimeter_mean</th><th scope=col>area_mean</th><th scope=col>smoothness_mean</th><th scope=col>compactness_mean</th><th scope=col>concavity_mean</th><th scope=col>concave.points_mean</th><th scope=col>...</th><th scope=col>texture_worst</th><th scope=col>perimeter_worst</th><th scope=col>area_worst</th><th scope=col>smoothness_worst</th><th scope=col>compactness_worst</th><th scope=col>concavity_worst</th><th scope=col>concave.points_worst</th><th scope=col>symmetry_worst</th><th scope=col>fractal_dimension_worst</th><th scope=col>X</th></tr></thead>
<tbody>
	<tr><td>  842302</td><td>M       </td><td>17.99   </td><td>10.38   </td><td>122.80  </td><td>1001.0  </td><td>0.11840 </td><td>0.27760 </td><td>0.3001  </td><td>0.14710 </td><td>...     </td><td>17.33   </td><td>184.60  </td><td>2019.0  </td><td>0.1622  </td><td>0.6656  </td><td>0.7119  </td><td>0.2654  </td><td>0.4601  </td><td>0.11890 </td><td>NA      </td></tr>
	<tr><td>  842517</td><td>M       </td><td>20.57   </td><td>17.77   </td><td>132.90  </td><td>1326.0  </td><td>0.08474 </td><td>0.07864 </td><td>0.0869  </td><td>0.07017 </td><td>...     </td><td>23.41   </td><td>158.80  </td><td>1956.0  </td><td>0.1238  </td><td>0.1866  </td><td>0.2416  </td><td>0.1860  </td><td>0.2750  </td><td>0.08902 </td><td>NA      </td></tr>
	<tr><td>84300903</td><td>M       </td><td>19.69   </td><td>21.25   </td><td>130.00  </td><td>1203.0  </td><td>0.10960 </td><td>0.15990 </td><td>0.1974  </td><td>0.12790 </td><td>...     </td><td>25.53   </td><td>152.50  </td><td>1709.0  </td><td>0.1444  </td><td>0.4245  </td><td>0.4504  </td><td>0.2430  </td><td>0.3613  </td><td>0.08758 </td><td>NA      </td></tr>
	<tr><td>84348301</td><td>M       </td><td>11.42   </td><td>20.38   </td><td> 77.58  </td><td> 386.1  </td><td>0.14250 </td><td>0.28390 </td><td>0.2414  </td><td>0.10520 </td><td>...     </td><td>26.50   </td><td> 98.87  </td><td> 567.7  </td><td>0.2098  </td><td>0.8663  </td><td>0.6869  </td><td>0.2575  </td><td>0.6638  </td><td>0.17300 </td><td>NA      </td></tr>
	<tr><td>84358402</td><td>M       </td><td>20.29   </td><td>14.34   </td><td>135.10  </td><td>1297.0  </td><td>0.10030 </td><td>0.13280 </td><td>0.1980  </td><td>0.10430 </td><td>...     </td><td>16.67   </td><td>152.20  </td><td>1575.0  </td><td>0.1374  </td><td>0.2050  </td><td>0.4000  </td><td>0.1625  </td><td>0.2364  </td><td>0.07678 </td><td>NA      </td></tr>
	<tr><td>  843786</td><td>M       </td><td>12.45   </td><td>15.70   </td><td> 82.57  </td><td> 477.1  </td><td>0.12780 </td><td>0.17000 </td><td>0.1578  </td><td>0.08089 </td><td>...     </td><td>23.75   </td><td>103.40  </td><td> 741.6  </td><td>0.1791  </td><td>0.5249  </td><td>0.5355  </td><td>0.1741  </td><td>0.3985  </td><td>0.12440 </td><td>NA      </td></tr>
</tbody>
</table>




```R
nrow(data)
```


569



```R
ncol(data)
```


33



```R
dim(data)
```


<ol class=list-inline>
	<li>569</li>
	<li>33</li>
</ol>




```R
names(data)
```


<ol class=list-inline>
	<li>'id'</li>
	<li>'diagnosis'</li>
	<li>'radius_mean'</li>
	<li>'texture_mean'</li>
	<li>'perimeter_mean'</li>
	<li>'area_mean'</li>
	<li>'smoothness_mean'</li>
	<li>'compactness_mean'</li>
	<li>'concavity_mean'</li>
	<li>'concave.points_mean'</li>
	<li>'symmetry_mean'</li>
	<li>'fractal_dimension_mean'</li>
	<li>'radius_se'</li>
	<li>'texture_se'</li>
	<li>'perimeter_se'</li>
	<li>'area_se'</li>
	<li>'smoothness_se'</li>
	<li>'compactness_se'</li>
	<li>'concavity_se'</li>
	<li>'concave.points_se'</li>
	<li>'symmetry_se'</li>
	<li>'fractal_dimension_se'</li>
	<li>'radius_worst'</li>
	<li>'texture_worst'</li>
	<li>'perimeter_worst'</li>
	<li>'area_worst'</li>
	<li>'smoothness_worst'</li>
	<li>'compactness_worst'</li>
	<li>'concavity_worst'</li>
	<li>'concave.points_worst'</li>
	<li>'symmetry_worst'</li>
	<li>'fractal_dimension_worst'</li>
	<li>'X'</li>
</ol>




```R
str(data)
```

    'data.frame':	569 obs. of  33 variables:
     $ id                     : int  842302 842517 84300903 84348301 84358402 843786 844359 84458202 844981 84501001 ...
     $ diagnosis              : Factor w/ 2 levels "B","M": 2 2 2 2 2 2 2 2 2 2 ...
     $ radius_mean            : num  18 20.6 19.7 11.4 20.3 ...
     $ texture_mean           : num  10.4 17.8 21.2 20.4 14.3 ...
     $ perimeter_mean         : num  122.8 132.9 130 77.6 135.1 ...
     $ area_mean              : num  1001 1326 1203 386 1297 ...
     $ smoothness_mean        : num  0.1184 0.0847 0.1096 0.1425 0.1003 ...
     $ compactness_mean       : num  0.2776 0.0786 0.1599 0.2839 0.1328 ...
     $ concavity_mean         : num  0.3001 0.0869 0.1974 0.2414 0.198 ...
     $ concave.points_mean    : num  0.1471 0.0702 0.1279 0.1052 0.1043 ...
     $ symmetry_mean          : num  0.242 0.181 0.207 0.26 0.181 ...
     $ fractal_dimension_mean : num  0.0787 0.0567 0.06 0.0974 0.0588 ...
     $ radius_se              : num  1.095 0.543 0.746 0.496 0.757 ...
     $ texture_se             : num  0.905 0.734 0.787 1.156 0.781 ...
     $ perimeter_se           : num  8.59 3.4 4.58 3.44 5.44 ...
     $ area_se                : num  153.4 74.1 94 27.2 94.4 ...
     $ smoothness_se          : num  0.0064 0.00522 0.00615 0.00911 0.01149 ...
     $ compactness_se         : num  0.049 0.0131 0.0401 0.0746 0.0246 ...
     $ concavity_se           : num  0.0537 0.0186 0.0383 0.0566 0.0569 ...
     $ concave.points_se      : num  0.0159 0.0134 0.0206 0.0187 0.0188 ...
     $ symmetry_se            : num  0.03 0.0139 0.0225 0.0596 0.0176 ...
     $ fractal_dimension_se   : num  0.00619 0.00353 0.00457 0.00921 0.00511 ...
     $ radius_worst           : num  25.4 25 23.6 14.9 22.5 ...
     $ texture_worst          : num  17.3 23.4 25.5 26.5 16.7 ...
     $ perimeter_worst        : num  184.6 158.8 152.5 98.9 152.2 ...
     $ area_worst             : num  2019 1956 1709 568 1575 ...
     $ smoothness_worst       : num  0.162 0.124 0.144 0.21 0.137 ...
     $ compactness_worst      : num  0.666 0.187 0.424 0.866 0.205 ...
     $ concavity_worst        : num  0.712 0.242 0.45 0.687 0.4 ...
     $ concave.points_worst   : num  0.265 0.186 0.243 0.258 0.163 ...
     $ symmetry_worst         : num  0.46 0.275 0.361 0.664 0.236 ...
     $ fractal_dimension_worst: num  0.1189 0.089 0.0876 0.173 0.0768 ...
     $ X                      : logi  NA NA NA NA NA NA ...
    


```R
summary(data)
```


           id            diagnosis  radius_mean      texture_mean  
     Min.   :     8670   B:357     Min.   : 6.981   Min.   : 9.71  
     1st Qu.:   869218   M:212     1st Qu.:11.700   1st Qu.:16.17  
     Median :   906024             Median :13.370   Median :18.84  
     Mean   : 30371831             Mean   :14.127   Mean   :19.29  
     3rd Qu.:  8813129             3rd Qu.:15.780   3rd Qu.:21.80  
     Max.   :911320502             Max.   :28.110   Max.   :39.28  
     perimeter_mean     area_mean      smoothness_mean   compactness_mean 
     Min.   : 43.79   Min.   : 143.5   Min.   :0.05263   Min.   :0.01938  
     1st Qu.: 75.17   1st Qu.: 420.3   1st Qu.:0.08637   1st Qu.:0.06492  
     Median : 86.24   Median : 551.1   Median :0.09587   Median :0.09263  
     Mean   : 91.97   Mean   : 654.9   Mean   :0.09636   Mean   :0.10434  
     3rd Qu.:104.10   3rd Qu.: 782.7   3rd Qu.:0.10530   3rd Qu.:0.13040  
     Max.   :188.50   Max.   :2501.0   Max.   :0.16340   Max.   :0.34540  
     concavity_mean    concave.points_mean symmetry_mean    fractal_dimension_mean
     Min.   :0.00000   Min.   :0.00000     Min.   :0.1060   Min.   :0.04996       
     1st Qu.:0.02956   1st Qu.:0.02031     1st Qu.:0.1619   1st Qu.:0.05770       
     Median :0.06154   Median :0.03350     Median :0.1792   Median :0.06154       
     Mean   :0.08880   Mean   :0.04892     Mean   :0.1812   Mean   :0.06280       
     3rd Qu.:0.13070   3rd Qu.:0.07400     3rd Qu.:0.1957   3rd Qu.:0.06612       
     Max.   :0.42680   Max.   :0.20120     Max.   :0.3040   Max.   :0.09744       
       radius_se        texture_se      perimeter_se       area_se       
     Min.   :0.1115   Min.   :0.3602   Min.   : 0.757   Min.   :  6.802  
     1st Qu.:0.2324   1st Qu.:0.8339   1st Qu.: 1.606   1st Qu.: 17.850  
     Median :0.3242   Median :1.1080   Median : 2.287   Median : 24.530  
     Mean   :0.4052   Mean   :1.2169   Mean   : 2.866   Mean   : 40.337  
     3rd Qu.:0.4789   3rd Qu.:1.4740   3rd Qu.: 3.357   3rd Qu.: 45.190  
     Max.   :2.8730   Max.   :4.8850   Max.   :21.980   Max.   :542.200  
     smoothness_se      compactness_se      concavity_se     concave.points_se 
     Min.   :0.001713   Min.   :0.002252   Min.   :0.00000   Min.   :0.000000  
     1st Qu.:0.005169   1st Qu.:0.013080   1st Qu.:0.01509   1st Qu.:0.007638  
     Median :0.006380   Median :0.020450   Median :0.02589   Median :0.010930  
     Mean   :0.007041   Mean   :0.025478   Mean   :0.03189   Mean   :0.011796  
     3rd Qu.:0.008146   3rd Qu.:0.032450   3rd Qu.:0.04205   3rd Qu.:0.014710  
     Max.   :0.031130   Max.   :0.135400   Max.   :0.39600   Max.   :0.052790  
      symmetry_se       fractal_dimension_se  radius_worst   texture_worst  
     Min.   :0.007882   Min.   :0.0008948    Min.   : 7.93   Min.   :12.02  
     1st Qu.:0.015160   1st Qu.:0.0022480    1st Qu.:13.01   1st Qu.:21.08  
     Median :0.018730   Median :0.0031870    Median :14.97   Median :25.41  
     Mean   :0.020542   Mean   :0.0037949    Mean   :16.27   Mean   :25.68  
     3rd Qu.:0.023480   3rd Qu.:0.0045580    3rd Qu.:18.79   3rd Qu.:29.72  
     Max.   :0.078950   Max.   :0.0298400    Max.   :36.04   Max.   :49.54  
     perimeter_worst    area_worst     smoothness_worst  compactness_worst
     Min.   : 50.41   Min.   : 185.2   Min.   :0.07117   Min.   :0.02729  
     1st Qu.: 84.11   1st Qu.: 515.3   1st Qu.:0.11660   1st Qu.:0.14720  
     Median : 97.66   Median : 686.5   Median :0.13130   Median :0.21190  
     Mean   :107.26   Mean   : 880.6   Mean   :0.13237   Mean   :0.25427  
     3rd Qu.:125.40   3rd Qu.:1084.0   3rd Qu.:0.14600   3rd Qu.:0.33910  
     Max.   :251.20   Max.   :4254.0   Max.   :0.22260   Max.   :1.05800  
     concavity_worst  concave.points_worst symmetry_worst   fractal_dimension_worst
     Min.   :0.0000   Min.   :0.00000      Min.   :0.1565   Min.   :0.05504        
     1st Qu.:0.1145   1st Qu.:0.06493      1st Qu.:0.2504   1st Qu.:0.07146        
     Median :0.2267   Median :0.09993      Median :0.2822   Median :0.08004        
     Mean   :0.2722   Mean   :0.11461      Mean   :0.2901   Mean   :0.08395        
     3rd Qu.:0.3829   3rd Qu.:0.16140      3rd Qu.:0.3179   3rd Qu.:0.09208        
     Max.   :1.2520   Max.   :0.29100      Max.   :0.6638   Max.   :0.20750        
        X          
     Mode:logical  
     NA's:569      
                   
                   
                   
                   

