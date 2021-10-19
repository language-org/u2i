# 2020_05_29

file: 2020_xx_xx.md  
topics: #nlp, #intent classification, #intent recognition, #intent detection, #slot filling  

* Public corpora: [3, 8]  
	* Industries: Banking, Computer, Air Travel, Movie industry, Food service industry  
	* `Clinc`
		* Sector: 
			* cross-domains: 10
				* Banking, Work, Meta, auto & commute, travel, home, utility, kitchen & dining, small talk, credit cards  			
		* task: task-oriented dialog, commands  
		* sample size: 23,700 instances
		* intents: 150 intents
		* multiple or single intent: single
		* 22,500 in-scope queries + 1,200 out-of-scope queries
		* mean sentence length: short (~10 words)
		* train input:   
			* format: .JSON
			* size: 2.38 MB
		* Generation: 
			* crowdsourcing from Quora, Wikipedia, etc.,...
		* url: [24]
		* paper: [25]
		* pros
			* Ecological relevance: 
    			* business relevant: many classes that emulate commercial systems env. 
				* many intents [13]
			* Contains in-scope & out-of-scope queries  
				* out-of-scope means not associated w/ any intents in the dataset
	* `HWU64` (`Liu 2019`):  
		+ date: 2019  
		* task: task-oriented dialog, commands  
		* 25,716 instances
		* 64 intents
		* 54 Entity types  				
		* license: CC BY-SA 3.0
		* corpus url: [26]
		* paper: [27]
		* generation: 
			* annotated by 3 students, mechanical turk
		* cross-domains: 
			* 21
			* alarm, audio, audiobook, calendar, cooking, datetime, email, game, general, IoT, lists, music, news, podcasts, general Q&A, radio, recommendations, social, food takeaway, transport, and weather  
		* unbalanced
		* train input
			* format: .csv
			* size: 5.6 MB			
		* pros
			* Ecological relevance: 
				* many intents, emulates commercial systems env. [13]
	* AskUbuntuCorpus [4, 9]  
		* Sector: Technology
		* Industry: Computer 
		* 162 instances of {Question, intent, entities, answer}  
		* 5 intents, 3 entities  
		* file format: .json  
		* description: tagged with mechanical turk  
		* license
		* limitations: 
			* small number of classes < 10 [13]
				* oversimplifies the task
				* poor ecological relevance
	* WebApplicationsCorpus [6, 9]  
		* Sector: Technology  
		* Industry: Computer  
		* 89 instances of {Question, intent, entities, answer}  
		* 8 intents, 3 entities  
		* file format: .json  
		* description: tagged with mechanical turk
		* license  
		* limitations  
			* small number of classes < 10 [13]
				* oversimplifies the task
				* poor ecological relevance				
	* ChatbotCorpus [5, 9]  
		* Sector: Transportation  
		* Industry: Air travel
		* 206 instances of {Question, intent, entities  
		* 2 intents, 5 entities  
		* file format: .json  
		* license  
		* limitations
			* small number of classes < 10 [13]
				* oversimplifies the task
				* poor ecological relevance				
	* ATIS  
		* Sector: Transportation  
		* Industry: Air travel  
		* url: [14, 19, 20]   
		* license  
		* description: airline travel information system
		* Initiator: byProduct of DARPA's ATIS corpus  
	* MIT movie  
		* Sector: Entertainment  
		* industry: Movie industry  
		* file format: .txt  
			* syntax: IOB  
		* url: [15]  
		* license  
	* MIT movie trivia
		* Sector: Entertainment  
		* industry: Movie industry  
		* train input format .txt  
			* syntax: IOB  
		* url: [16]  
		* license  
	* MIT restaurant  
		* Sector: Entertainment  
		* industry: Food service industry  
		* url: [17]  
		* license  
	* Snips  
		* task: task-oriented dialog
		* 14,484 query instances
		* 7 intents
		* 70 instances per intent  
		* Generation: crowdsourcing  
		* url: [23]  
		* limitations  
			* small number of intents < 10 [13]
				* oversimplifies the task
				* poor ecological relevance				
	* [12]  
	* [11, 12]  
	* Banking77  
		* Sector: Finance  
		* Industry: Banking  
		* task: task-oriented dialog
		* 13,083 {question, intents}  
		* 77 intents  
		* train input format: .csv  
		* url: [13]  
		* corpus url [30]
		* license  
		* pros
			* Ecological relevance: 
				* many intents, emulates commercial systems env. [13]
	* Facebook Task-oriented dialogue
		* 
		*
	* Sweet-Home dataset [21]
		* Sector: Technology  
		* Industry: Domotics  
		* license: only for research 
	* TREC 
		* date: 2002
		* task: question classification  
		* 6000 query instances  
		* 6 intents (labels), 50 level-2 labels    
		* mean sentence length: short (10 words)  
		* vocabulary: 8700 unique words
		* corpus url: [27]
		* paper: [29]   
	* others
    	* (31)

to check
* https://appen.com/datasets/audio-recording-and-transcription-for-medical-scenarios/?amp  

## References

[1] https://chatbotslife.com/know-your-intent-sota-results-in-intent-classification-8e1ca47f364c   
[3] https://github.com/sebischair/NLU-Evaluation-Corpora  
[4] https://github.com/sebischair/NLU-Evaluation-Corpora/blob/master/AskUbuntuCorpus.json  
[5] https://github.com/sebischair/NLU-Evaluation-Corpora/blob/master/ChatbotCorpus.json  
[6] https://github.com/sebischair/NLU-Evaluation-Corpora/blob/master/WebApplicationsCorpus.json  
[7] https://towardsdatascience.com/multi-label-intent-classification-1cdd4859b93
[8] https://www.aclweb.org/anthology/2020.lrec-1.873.pdf
[9] Braun, D., Mendez, A. H., Matthes, F., & Langen, M. (2017, August). Evaluating natural language understanding services for conversational question answering systems. In Proceedings of the 18th Annual SIGdial Meeting on Discourse and Dialogue (pp. 174-185).  
[11] Larson, S., Mahendran, A., Peper, J. J., Clarke, C., Lee, A., Hill, P., Kummerfeld, J. K., Leach, K., Laurenzano, M. A., Tang, L., and Mars, J. (2019). An evaluation dataset for intent classification and out-of-scope prediction. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP)    
[12] Data Query Language and Corpus Tools for Slot-Filling and Intent Classification Data     
[13] Casanueva, I., Temčinas, T., Gerz, D., Henderson, M., & Vulić, I. (2020). Efficient Intent Detection with Dual Sentence Encoders. arXiv preprint arXiv:2003.04807.  
[14] https://github.com/sz128/slot_filling_and_intent_detection_of_SLU/tree/master/data/atis-2  
[15] https://github.com/sz128/slot_filling_and_intent_detection_of_SLU/tree/master/data/MIT_corpus/movie_eng  
[16] https://github.com/sz128/slot_filling_and_intent_detection_of_SLU/tree/master/data/MIT_corpus/movie_  trivia10k13  
[17] https://github.com/sz128/slot_filling_and_intent_detection_of_SLU/blob/master/data/MIT_corpus/restaurant/train   
[18] https://github.com/sz128/slot_filling_and_intent_detection_of_SLU/tree/master/data/multilingual_task_oriented_data/en  
[19] https://github.com/yvchen/JointSLU  
[20] https://github.com/microsoft/CNTK/tree/master/Examples/LanguageUnderstanding   
[21] http://sweet-home-data.imag.fr    
[22] https://www.microsoft.com/en-us/research/wp-content/uploads/2010/12/SLT10.pdf  
[23] https://github.com/snipsco/nlu-benchmark/tree/master/2017-06-custom-intent-engines  
[24] https://github.com/clinc/oos-eval  
[25] Stefan Larson et al., “An Evaluation Dataset for Intent Classification and Out-of-Scope Prediction,” in Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP) (EMNLP-IJCNLP 2019, Hong Kong, China: Association for Computational Linguistics, 2019), 1311–1316, https://doi.org/10.18653/v1/D19-1131.  
[26] https://github.com/xliuhw/NLU-Evaluation-Data  
[27] Liu, X., Eshghi, A., Swietojanski, P., & Rieser, V. (2019). Benchmarking natural language understanding services for building conversational agents. arXiv preprint arXiv:1903.05566.  
[28] https://pytorchnlp.readthedocs.io/en/latest/_modules/torchnlp/datasets/trec.html  
[29] Li, X., & Roth, D. (2002). Learning question classifiers. In COLING 2002: The 19th International Conference on Computational Linguistics, https://dl.acm.org/doi/pdf/10.3115/1072228.1072378    
[30] https://github.com/PolyAI-LDN/task-specific-datasets  
(31) https://huggingface.co/datasets?filter=task_ids:intent-classification    

## Workload 

* 1.5 days