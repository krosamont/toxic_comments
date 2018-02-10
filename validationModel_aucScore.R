library(tidyverse)
library(keras)
if (!require("caret")) install.packages("caret")
library(caret) 
if (!require("ROSE")) install.packages("ROSE")
library(ROSE) 

#You can read the all process or directly run the results from line 104 to 115 
#your path

comments = readRDS("data/comments.rds") %>%
        mutate(id_num = 1:n())
comments_clean = readRDS("data/comments_clean.rds")
comments_stem = readRDS( "data/comments_stem.rds")
comments_tf_idf = readRDS("data/comments_tf_idf.rds")
set.seed(3456)
trainIndex = createDataPartition(comments$id_num, p = .8, list = FALSE)

comments_test = comments[-trainIndex,]
comments_tf_idf_test = comments_tf_idf[-trainIndex,]
comments_clean_test = comments_clean[-trainIndex,]
comments_stem_test = comments_stem[-trainIndex,]

max_words = 20000
maxl = 300


t_roc = NULL
t_auc = NULL


roc_auc = function(x, type){
        if(type=="toxic"){ 
                roc = roc.curve(x$toxic, x$pred.toxic, plotit = FALSE)
        } else if( type == "severe_toxic"){
                roc = roc.curve(x$severe_toxic, x$pred.severe_toxic, plotit = FALSE)
        } else if(type == "obscene"){
                roc = roc.curve(x$obscene, x$pred.obscene, plotit = FALSE)
        } else if(type== "threat"){
                roc = roc.curve(x$threat, x$pred.threat, plotit = FALSE)
        } else if(type == "insult"){
                roc = roc.curve(x$insult, x$pred.insult, plotit = FALSE)
        } else if(type== "id_hate"){
                roc = roc.curve(x$identity_hate, x$pred.identity_hate, plotit = FALSE)
        }
        
        
        res = list(auc=data.frame(auc=roc[[2]], model=x$model, type=type)[1,], roc=data.frame(fp=roc[[3]], tp=roc[[4]], type=type, model=rep(x$model, length(roc[[4]]))
        ))
        return(res)
}


for(i in 1:4){
        if(i==1){final_data = comments_test}
        else if(i==2){final_data = comments_tf_idf_test}
        else if(i==3){final_data = comments_clean_test}
        else{final_data = comments_stem_test}
        
        for(k in 1:2){
                
                wordseq = text_tokenizer(num_words = max_words) %>%
                        fit_text_tokenizer(final_data$comment_text)
                
                
                # The 1 corresponds to the book methodology
                sequences = texts_to_sequences(wordseq, final_data$comment_text )
                test_data = pad_sequences(sequences, maxlen = maxl)
                model = load_model_hdf5(paste0("model/checkpoints_model",k,"_",i,".h5"))
                
                pred = model %>%
                        predict_proba(test_data) %>%
                        cbind(id=comments_test$id, comments_test[3:8]) %>%
                        mutate(model=paste0("model",k,"_",i))
                
                names(pred)[1:6] = c("pred.toxic", "pred.severe_toxic", "pred.obscene", "pred.threat","pred.insult", "pred.identity_hate")
                
                t = c("toxic", "severe_toxic", "obscene", "threat", "insult", "id_hate")
                
                
                auc_model = map(t,function(x){roc_auc(pred,x)}) %>%
                        map("auc") %>%
                        reduce(rbind)
                t_auc = rbind(t_auc, auc_model)
                
                roc_model = map(t,function(x){roc_auc(pred,x)}) %>%
                        map("roc") %>%
                        reduce(rbind)
                t_roc = rbind(t_roc, roc_model) %>%
                        group_by_all() %>%
                        filter(row_number()==1)
                cat(paste0("Done for model ",k, " with data ",i, " at ", Sys.time(), "\n" ))
        }
        #saveRDS(t_auc, "auc_total3.rds")
        #saveRDS(t_roc, "roc_total3.rds")
        
}


auc = readRDS("data/auc_total.rds")
res = auc %>%
        group_by(model) %>%
        summarise(average_auc= mean(auc)) %>%
        arrange(desc(average_auc))

#saveRDS(roc, "roc_total.rds")

roc = readRDS("data/roc_total.rds")
ggplot(roc, aes(x=fp, y=tp, colour = model)) + labs(x ="FALSE POSITIVE", y = "TRUE POSITIVE") +
        geom_line() + facet_wrap(~type, ncol = 2, scales = "free")