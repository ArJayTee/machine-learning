# Machine Learning Engineer Nanodegree
## Capstone Proposal
Richard J. Taylor, PMP
October 9, 2018

## Proposal

Using News to Predict Stock Movements

https://www.kaggle.com/c/two-sigma-financial-news/data 


### Domain Background
_(approx. 1-2 paragraphs)_

Accurately predicting stock price performance is akin to the alchemy of turning lead into gold. There are those that have the ability to see trends, invest in correlation to those trends and make money. However, accurate stock prediction is still an arduous if not impossible task. However, using machine learning one may gleam some insight into why these fluctuations occur.
This project will use data from specific days stock trades, compile this and compare it with a series of news articles which are published on the same date, data will also include a sentiment analysis. Using this data the algorithm will attempt to predict stock predictions. 

### Problem Statement
_(approx. 1 paragraph)_

This project will help identify whether machine learning techniques can predict stock market flucuations based on the current market state and news articles pertaining to specific types of stocks. Predictions will be stated in range [-1.0, 1.0], where the positive value is an increase in value and negative value is a decrease in value over a 10 trading day period. 

### Datasets and Inputs
_(approx. 2-3 paragraphs)_
In this section, the dataset(s) and/or input(s) being considered for the project should be thoroughly described, such as how they relate to the problem and why they should be used. Information such as how the dataset or input is (was) obtained, and the characteristics of the dataset or input, should be included with relevant references and citations as necessary It should be clear how the dataset(s) or input(s) will be used in the project and whether their use is appropriate given the context of the problem.


The datasets used are provided by Intrinio via Kaggle.com and include the following: 
"1. Market data (2007 to present) provided by Intrinio - contains financial market information such as opening price, closing price, trading volume, calculated returns, etc."
The marketdata_sample data includes: 

•	time(datetime64[ns, UTC]) - the current time (in marketdata, all rows are taken at 22:00 UTC)

•	assetCode(object) - a unique id of an asset

•	assetName(category) - the name that corresponds to a group of assetCodes. These may be "Unknown" if the corresponding assetCode does not have any rows in the news data.

•	universe(float64) - a boolean indicating whether or not the instrument on that day will be included in scoring. This value is not provided outside of the training data time period. The trading universe on a given date is the set of instruments that are avilable for trading (the scoring function will not consider instruments that are not in the trading universe). The trading universe changes daily.

•	volume(float64) - trading volume in shares for the day

•	close(float64) - the close price for the day (not adjusted for splits or dividends)

•	open(float64) - the open price for the day (not adjusted for splits or dividends)

•	returnsClosePrevRaw1(float64) - see returns explanation above

•	returnsOpenPrevRaw1(float64) - see returns explanation above

•	returnsClosePrevMktres1(float64) - see returns explanation above

•	returnsOpenPrevMktres1(float64) - see returns explanation above

•	returnsClosePrevRaw10(float64) - see returns explanation above

•	returnsOpenPrevRaw10(float64) - see returns explanation above

•	returnsClosePrevMktres10(float64) - see returns explanation above

•	returnsOpenPrevMktres10(float64) - see returns explanation above

•	returnsOpenNextMktres10(float64) - 10 day, market-residualized return. This is the target variable used in competition scoring. The market data has been filtered such that returnsOpenNextMktres10 is always not null.


2. "News data (2007 to present) Source: Thomson Reuters - contains information about news articles/alerts published about assets, such as article details, sentiment, and other commentary." 
Link to the Kaggle data - https://www.kaggle.com/dster/two-sigma-news-official-getting-started-kernel/data

The news_sample data contains:

•	time(datetime64[ns, UTC]) - UTC timestamp showing when the data was available on the feed (second precision)

•	sourceTimestamp(datetime64[ns, UTC]) - UTC timestamp of this news item when it was created

•	firstCreated(datetime64[ns, UTC]) - UTC timestamp for the first version of the item

•	sourceId(object) - an Id for each news item

•	headline(object) - the item's headline

•	urgency(int8) - differentiates story types (1: alert, 3: article)

•	takeSequence(int16) - the take sequence number of the news item, starting at 1. For a given story, alerts and articles have separate sequences.

•	provider(category) - identifier for the organization which provided the news item (e.g. RTRS for Reuters News, BSW for Business Wire)

•	subjects(category) - topic codes and company identifiers that relate to this news item. Topic codes describe the news item's subject matter. These can cover asset classes, geographies, events, industries/sectors, and other types.

•	audiences(category) - identifies which desktop news product(s) the news item belongs to. They are typically tailored to specific audiences. (e.g. "M" for Money International News Service and "FB" for French General News Service)

•	bodySize(int32) - the size of the current version of the story body in characters

•	companyCount(int8) - the number of companies explicitly listed in the news item in the subjects field

•	headlineTag(object) - the Thomson Reuters headline tag for the news item

•	marketCommentary(bool) - boolean indicator that the item is discussing general market conditions, such as "After the Bell" summaries

•	sentenceCount(int16) - the total number of sentences in the news item. Can be used in conjunction with firstMentionSentence to determine the relative position of the first mention in the item.

•	wordCount(int32) - the total number of lexical tokens (words and punctuation) in the news item

•	assetCodes(category) - list of assets mentioned in the item

•	assetName(category) - name of the asset

•	firstMentionSentence(int16) - the first sentence, starting with the headline, in which the scored asset is mentioned.
 
  o	1: headline
  
  o	2: first sentence of the story body
  
  o	3: second sentence of the body, etc
  
  o	0: the asset being scored was not found in the news item's headline or body text. As a result, the entire news item's text (headline + body) will be used to determine the sentiment score.

•	relevance(float32) - a decimal number indicating the relevance of the news item to the asset. It ranges from 0 to 1. If the asset is mentioned in the headline, the relevance is set to 1. When the item is an alert (urgency == 1), relevance should be gauged by firstMentionSentence instead.

•	sentimentClass(int8) - indicates the predominant sentiment class for this news item with respect to the asset. The indicated class is the one with the highest probability.

•	sentimentNegative(float32) - probability that the sentiment of the news item was negative for the asset

•	sentimentNeutral(float32) - probability that the sentiment of the news item was neutral for the asset

•	sentimentPositive(float32) - probability that the sentiment of the news item was positive for the asset

•	sentimentWordCount(int32) - the number of lexical tokens in the sections of the item text that are deemed relevant to the asset. This can be used in conjunction with wordCount to determine the proportion of the news item discussing the asset.

•	noveltyCount12H(int16) - The 12 hour novelty of the content within a news item on a particular asset. It is calculated by comparing it with the asset-specific text over a cache of previous news items that contain the asset.

•	noveltyCount24H(int16) - same as above, but for 24 hours

•	noveltyCount3D(int16) - same as above, but for 3 days

•	noveltyCount5D(int16) - same as above, but for 5 days

•	noveltyCount7D(int16) - same as above, but for 7 days

•	volumeCounts12H(int16) - the 12 hour volume of news for each asset. A cache of previous news items is maintained and the number of news items that mention the asset within each of five historical periods is calculated.

•	volumeCounts24H(int16) - same as above, but for 24 hours

•	volumeCounts3D(int16) - same as above, but for 3 days

•	volumeCounts5D(int16) - same as above, but for 5 days

•	volumeCounts7D(int16) - same as above, but for 7 days


### Solution Statement
_(approx. 1 paragraph)_

In this section, clearly describe a solution to the problem. The solution should be applicable to the project domain and appropriate for the dataset(s) or input(s) given. Additionally, describe the solution thoroughly such that it is clear that the solution is quantifiable (the solution can be expressed in mathematical or logical terms) , measurable (the solution can be measured by some metric and clearly observed), and replicable (the solution can be reproduced and occurs more than once).

### Benchmark Model
_(approximately 1-2 paragraphs)_

In this section, provide the details for a benchmark model or result that relates to the domain, problem statement, and intended solution. Ideally, the benchmark model or result contextualizes existing methods or known information in the domain and problem given, which could then be objectively compared to the solution. Describe how the benchmark model or result is measurable (can be measured by some metric and clearly observed) with thorough detail.

### Evaluation Metrics
_(approx. 1-2 paragraphs)_

In this section, propose at least one evaluation metric that can be used to quantify the performance of both the benchmark model and the solution model. The evaluation metric(s) you propose should be appropriate given the context of the data, the problem statement, and the intended solution. Describe how the evaluation metric(s) are derived and provide an example of their mathematical representations (if applicable). Complex evaluation metrics should be clearly defined and quantifiable (can be expressed in mathematical or logical terms).

### Project Design
_(approx. 1 page)_

In this final section, summarize a theoretical workflow for approaching a solution given the problem. Provide thorough discussion for what strategies you may consider employing, what analysis of the data might be required before being used, or which algorithms will be considered for your implementation. The workflow and discussion that you provide should align with the qualities of the previous sections. Additionally, you are encouraged to include small visualizations, pseudocode, or diagrams to aid in describing the project design, but it is not required. The discussion should clearly outline your intended workflow of the capstone project.

-----------

**Before submitting your proposal, ask yourself. . .**

- Does the proposal you have written follow a well-organized structure similar to that of the project template?
- Is each section (particularly **Solution Statement** and **Project Design**) written in a clear, concise and specific fashion? Are there any ambiguous terms or phrases that need clarification?
- Would the intended audience of your project be able to understand your proposal?
- Have you properly proofread your proposal to assure there are minimal grammatical and spelling mistakes?
- Are all the resources used for this project correctly cited and referenced?
