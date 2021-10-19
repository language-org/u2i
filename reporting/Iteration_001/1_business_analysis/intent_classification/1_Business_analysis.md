file: Business_analysis.md


## Use case reviews

* Highest-value:

	* `Intent classification module` 
		* Business value
			* critical for chatbot development  
			* document classification  
			* customer intent about a product  
			* health care 
				* Purpose of a patient visit to the doctor [1]
				* Medical query intent classification [2, 3]				
				* disease inference 
		* requirements (BEN M.):  
			* first approach should not be SOTA.. it is often not worth the cost and difficult to improve
			* is the model lighweight and heavy (many GB) ?  
			* We can upload our own corpus (text / intent) to train the model
			* Does the user obtain their own version of the model in this case?
			* Can we download the trained model to use for a client project?
			* Can you find a public corpus of question / intent to try it out on?
			* Compare to ML pipeline text --> vectorized text (TF/IDF or w2v or ELMO) --> sklearn classifer
			* Another interesting question would be to incorporate multilabel classifier (one question can have multiple intents)
			* check Google api
		* stay up-to-date with Oleg and possibly collaborate  

* Other possible use cases:

	* `Demonstrator on RASA` : [Priority !]
		* how to use it 
		* didactic

	* `Demonstrator for sentiment analysis` 
		* notebook to demonstrate how to do this   
		sentiment dictionary - polarity  couple of days
		notebook on how to download imdb datasets 

	* `NER` component


## References

[zotero_ok]
[1] https://medium.com/@sean.sodha/ibm-watson-nlc-intent-classification-39f7ac83b079  
[2] Cai, R., Zhu, B., Ji, L., Hao, T., Yan, J., & Liu, W. (2017, November). An CNN-LSTM attention approach to understanding user query intent from online health communities. In 2017 IEEE International Conference on Data Mining Workshops (ICDMW) (pp. 430-437). IEEE.  
	* Microsoft research et al.
	* intent taxonomy and labeling the dataset for training and validating the prediction model [2]
[3] Li, Y., Liu, C., Du, N., Fan, W., Li, Q., Gao, J., ... & Wu, H. (2016). Extracting medical knowledge from crowdsourced question answering website. IEEE Transactions on Big Data.
	* Baidu Research Big Data Lab et al. 
	* unsupervised learning 