# Using Predictive Modeling with Logistic Regression to Determine the Success of a Coaching Change in the NBA

(Note: the file structure and code in this project is rather unorganized or generally not well-written. Apologies for that! This was my first modeling project so I wasn't aware of proper documentation conventions, file structure formatting, or how to use pandas more efficiently. It was a fun learning experience though which is why I've kept it public.)

In the National Basketball Association (NBA), league executives are continually making decisions on how to best improve their team, including hiring new head coaches. The rationale behind firing and hiring coaches is generally that a new coach can help the team win more games compared to the previous coach, and through steady improvement, ultimately win an NBA title. The purpose of this paper is to determine the accuracy with which we can predict whether an NBA coaching change will be successful by comparing teams’ change in win-loss percentage after hiring a new coach, with a successful coaching hire being defined as the team improving their win-loss percentage from the previous coach to the current coach. I began by exporting a CSV file of each NBA team’s franchise index on Basketball Reference to create a master data frame consisting of every coaching change since the 1979-80 NBA season (≥ 15 games coached), identifying 402 coaching changes since 1979 and calculating each coach’s respective change in win-loss percentage compared to the previous coach, accounting for midseason hirings. Using a web-scraping tool sourced from Github, I retrieved 19 features to regress each change in win-loss percentage onto, eliminating multicollinear features ( |r| ≥ 0.75) and features that were not correlated ( |r| < 0.1) with change in win-loss percentage. After one-hot encoding each coaching change as successful (1) or unsuccessful (0), I used multiple logistic regression to produce a confusion matrix, finding that my binary classification model was able to predict whether a coaching change was successful with about 68.6 percent accuracy. Following the initial construction of this model, I separated each coaching change by whether the coach was a mid-season hire or not. In addition to finding no statistically significant difference in a team’s change in win-loss percentage regardless of whether the coach was a mid-season hire or not (p > 0.1), I found that my multiple logistic regression model was able to predict whether a full-season hire would be successful with 76.2 percent accuracy, whereas it predicted the success of a mid-season hire with 63.2 percent accuracy. This paper seeks to interpret the results of these findings in the context of team management’s hiring decisions.