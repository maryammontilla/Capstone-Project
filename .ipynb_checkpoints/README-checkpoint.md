# Hair Loss Prediction!

## Problem Statement
- Given data from the Kaggle Hair Loss Dataset, I analyzed different features which correlate to hair loss.
- The dataset is intended for exploratory data analysis, modeling, and predictive analytics tasks aimed at understanding the relationship between various factors and the likelihood of baldness in individuals.
## Data Dictionary
**Genetics**: Indicates whether the individual has a family history of baldness (Yes/No)(1/0).

**Hormonal Changes**: Indicates whether the individual has experienced hormonal changes (Yes/No)(1/0).

**Medical Conditions**: Lists specific medical conditions that may contribute to baldness, such as Alopecia Areata, Thyroid Problems, Scalp Infection, Psoriasis, Dermatitis, etc.

**Medications & Treatments**: Lists medications and treatments that may lead to hair loss, such as Chemotherapy, Heart Medication, Antidepressants, Steroids, etc.

**Nutritional Deficiencies**: Lists nutritional deficiencies that may contribute to hair loss, such as Iron deficiency, Vitamin D deficiency, Biotin deficiency, Omega-3 fatty acid deficiency, etc.

**Stress**: Indicates the stress level of the individual (Low/Moderate/High).

**Age**: Represents the age of the individual.

**Poor Hair Care Habits**: Indicates whether the individual practices poor hair care habits (Yes/No)(1/0).

**Environmental Factors**: Indicates whether the individual is exposed to environmental factors that may contribute to hair loss (Yes/No)(1/0).

**Smoking**: Indicates whether the individual smokes (Yes/No)(1/0).

**Weight Loss**: Indicates whether the individual has experienced significant weight loss (Yes/No)(1/0).

**Baldness (Target)**: Binary variable indicating the presence (1) or absence (0) of baldness in the individual.

## Executive Summary

### Data Cleaning Steps
When cleaning out the data, I checked to see if there was any null values that might skew the results of our data analysis and found that there were some 'No Data' values in the columns 'Nutritional Deficiencies ', 'Medical Conditions', and 'Medications & Treatments'. I decided that these values might be useful as they may be values that indicate no medical conditions, nutritional deficiencies, and medications/treatments. Because of this, I did not remove these values.
I then took a look at the categorical features in the dataset by looking at the dtypes of our columns. I was able to discover columns which contained 'Yes'/'No' values and converted them from a string into 0/1 binary numerical values. This step was necessary so that I can get a better visualization of the features that correlate to our target variable of hair loss. I also dummified the categorical data to fit those features in my model for the best predictions. Once these columns and values were cleaned up, I checked to see if there were any additional outliers or null values. After confirming the data has been cleaned up, I then moved on to visualizing the data!

### Key Visualizations

#### Visualization 1: Medical Conditions & Hair Loss
This chart illustrates individuals with the presence of hair loss by medical conditions. Using the information from the individuals in our dataset, 57% of individuals with Alopecia Areata experience hair loss, 56.8% of those with Seborrheic Dermatitis experince hair loss, and 56.1% of those with Androgenetic Alopecia experience hair loss.
While 57.3% of those with no medical conditions do not experience hair loss. 
There seems to be a correlation of hair loss being more prevalent in those who suffer from medical conditions such as Alopecia Areata, Androgenetic Alopecia, and Sebhorrheic Dermatitis which makes sense since Sebhorrheic Dermatitis causes itchiness of the scalp which can cause irritation to the hair follicles resulting in hair loss. Alopecia is also a medical condition which results in the destruction or heightened sensitivity of hair follicles causing permanent hair loss as well. 

![Visualization 1](submissions/medconditions.png)

#### Visualization 2: Hair Loss & Medications/Treatments
In this chart, The majority of individuals taking medications/treaments such as Steroids, Antibiotics, Chemotherapy, and Heart Medication experience hair loss while the majority of individuals taking Immunomodulators, Blood Pressure Medication, and Antifungal Creams experience no hair loss. Those who are not on any medications/treatments ('No Data') have an even 50/50 split of experiencing hair loss or not.

![Visualization 2](submissions/medic_treatmnts.png)

#### Visualization 3: Genetics & Hair Loss
In this visualization, 51.7% of individuals with a family history of baldness also end up experiencing hair loss whereas only 47.6% of those individuals who do not have a family history of baldness experience hair loss. This shows a slight positive correlation of how genetics play a role in hair loss.

![Visualization 3](submissions/genetics.png)

#### Visualization 4: Hormonal Changes & Hair Loss
In this visualization, 50.1% of individuals experiencing hormonal changes also experience hair loss while 49.4% of those who do not experience hormonal changes also experience hair loss. This visualization is interesting as there is not a strong correlation between hormonal changes and hair loss. 

![Visualization 4](submissions/hormonalchanges.png)

#### Visualization 5: Nutritional Deficiencies & Hair Loss
 54.8% of those with a magnesium defficiency experience hair loss, 52.5% of those with no deficiency also experince hair loss, 52.2% of those with protein deficiency experience hair loss compared to those who are deficient in omega-3 fatty acid, vitamin E, and Biotin.
 
![Visualization 5](submissions/nutrition_def.png)

#### Visualization 6: Stress & Hair Loss
This visualiztion is interesting because you would assume that those individuals experincing high levels of stress would experience more hair loss; However, from the data in our dataset it appears that only 48.6% of those experiencing high levels of stress and low levels of stress experience hair loss while 51.9% of individuals with moderate levels of stress experience hair loss. 

![Visualization 6](submissions/stress.png)

#### Visualization 7: Age & Hair Loss
Those with no presence of hair loss have a higher age on average than those individuals in the dataset who did experience hair loss. You can also visualize with the orange line the presence of hair loss among individuals by age and it reveals that the majority of individuals in the dataset experiencing hair loss are around 27-37 years old.

![Visualization 7](submissions/age.png)
![Visualization 7](submissions/age2.png)

#### Visualization 8: Poor Hair Care Habits & Hair Loss
In this visualization, there appears to be only 47.8% of individuals with poor hair care habits experiencing hair loss while 51.7% of those who do not have poor hair care habits actually experience hair loss. 

![Visualization 8](submissions/poorhairhabits.png)

#### Visualization 9: Environmental Factors & Hair Loss
This visualization reveals that only 48.8% of individuals who are exposed to environmental factors that might impact hair loss actually do experience hair loss while 50.7% of those individuals who are not exposed to those environmental factors are experiencing hair loss. 

![Visualization 9](submissions/envirofactors.png)

#### Visualization 10: Smoking & Hair Loss
47% of individuals in the dataset who are smokers experience hair loss while 52.7% of those who do not smoke experience hair loss. This visualization reveals that there is not a strong correlation between smoking and hair loss among the data collected in our dataset.

![Visualization 10](submissions/smoking.png)

#### Visualization 11: Weight Loss & Hair Loss
This visualization shows that the majority (52.1%) of those who experience weight loss also experience hair loss. while the majority of those who do not experience weight loss (52.4%) also do not experience hair loss. This reveals that there may be a correlation between weight loss and hair loss. 

![Visualization 11](submissions/weightloss.png)

## Modeling

###
## Conclusions/Recommendations
From these findings, it is evident that there are some correlations between hair loss and other features such as