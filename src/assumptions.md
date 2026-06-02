### Model Assumptions

This analysis estimates speed/watt differences between bikes by training a model on Bike A efforts and measuring how Bike B efforts deviate from that prediction. It is designed to be simple and interpretable, not perfect. This is kind of a simplified double machine learning causal inference. I do not think I can do a truly rigorous causal inference here because I do not want to enforce that the 2 bikes were ridden in the same time period, and without enforcing that I will always have too many confounding factors.

So this is more of a residual analysis that I am hoping lines up with how I feel on my bike.


#### Outlier filtering
Outlier filtering does a lot of heavy lifting here. I am assuming that with the chosen filters in data cleaning step, we are able to remove a whole bunch of confounders. Drafting, tri bar usage, carrying heavy backpack, etc. So I would like to think that the outlier filtering step makes a lot of these assumptions a bit more reasonable. But there is no denying that there are a lot of assumptions in this analysis.

#### Broadly Speaking: Transferability of the Counterfacftual
In this analysis I am assuming my model is perfect and the speed I say I wouldve gone on the other bike is the real speed. This is likely not true because my model doesnt account for everything. 

#### Fitness
I am basically assuming that fitness is entirely irrelevant because being more fit just means you make more watts and I have watt numbers. This is not entirely true. For example when more fit I can put out more watts in my aero position. So this is a stretch.

#### Braking
I have no data on how much time did I spend on the brakes. This is a large factor in how fast you go on a bike that I am not representing. I think accounting for curvy-ness of route could account for some of this, and maybe outlier filtering will get some of it as well. But it is a big blind spot.

#### Sample Size
I dont enforce a sample size because I dont want to wait until I have more data. And especially with all my assumptions, a large N would be much better for this project.

#### Average grade issue
I have average and max grade for segments, but this isnt as representative as I would like it to be of a segment. For example a segment that goes up down up down and averages to average grade 0.

#### Traffic
My model does not account for traffic stopages.

#### Wind
This is a big factor that I am not currently modeling - though I plan to bring it in using a weather api