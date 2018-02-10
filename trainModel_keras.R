library(tidyverse)
library(keras)
if (!require("tidytext")) install.packages("tidytext")
library(tidytext)
if (!require("qdapRegex")) install.packages("qdapRegex")
library(qdapRegex) 
if (!require("stringr")) install.packages("stringr")
library(stringr)
if (!require("tm")) install.packages("tm")
library(tm) 
if (!require("caret")) install.packages("caret")
library(caret) 

comments = readRDS("data/comments.rds")
source("useful_functions.R")
comments = comments %>%
        mutate(comment_text = tolower(preclean(comment_text)), id_num=1:nrow(comments)) %>%
        mutate(comment_text = sms_vocab(comment_text))
#saveRDS(comments, "data/comments_preclean.rds")

toxic = comments %>%
        filter(toxic==1) 
sev_toxic = comments %>%
        filter(severe_toxic==1)
obscene = comments %>%
        filter(obscene==1)
threat = comments %>%
        filter(threat==1)
insult = comments %>%
        filter(insult==1)
id_hate = comments %>%
        filter(identity_hate==1)
none = comments %>%
        filter(toxic==0 & severe_toxic==0 & obscene==0 & threat==0 & insult==0 & identity_hate==0)

docs= list(none=none, toxic=toxic, sev_toxic=sev_toxic, obscene=obscene, threat=threat, insult=insult,  id_hate=id_hate)

distrib = docs %>%
        map(nrow) %>%
        reduce(rbind) %>%
        as.data.frame(row.names = FALSE)


distrib = distrib %>%
        rename(count=V1) %>%
        mutate( percent = round(100*count/nrow(comments),2), 
                type = c("none", names(comments)[3:8]) )

ggplot(distrib, aes(x = reorder(type, -percent), count, fill = type)) +
        geom_col(show.legend = FALSE) +
        labs(x = NULL, y = "count comments") +
        geom_text(aes(label=paste0(percent,"%")), vjust=-0.3, color="black", size=3.5) 

wgt = readRDS("data/glove.6B/glove_6B_50d.rds")
wgt = wgt %>%
        mutate(word=gsub("[[:punct:]]"," ", rm_white(word) ))
dic_words = wgt$word

#unknwon_words(comments$comment_text[6:9])
init=Sys.time()
unknown = docs %>% 
        map("comment_text") %>% 
        map(str_split, pattern = " ") %>% 
        map(unlist) 

unknown = unknown %>% 
        map(table) %>%
        map(as.data.frame) %>%
        map2(names(docs), ~mutate(.x,type=.y)) %>%
        reduce(rbind) 

unknown = unknown %>% 
        rename(word=Var1) %>%
        filter(!word  %in% wgt$word)
#saveRDS(unknown, "unknown_words.rds")


char_comments = docs %>%
        map("comment_text") %>%
        map(length_comments) %>%
        map2(names(docs),~mutate(.x, type=.y)) %>%
        reduce(rbind)
#saveRDS(char_comments,"length_comments.rds")

#To remove the outlier that has more than 1500 words
ggplot(char_comments %>% filter(l_comment<1500), aes(unique_per_l, l_comment))+
        labs(x = "numbe of words", y = "unique words") + geom_point(aes(colour = type, alpha=0.8))

ggplot(char_comments, aes(x = type, y = unique_per_l, fill = type)) +
        geom_boxplot(alpha=0.7, outlier.alpha = 0) +
        scale_y_continuous(name = "Unique words per word comments",
                           #breaks = seq(0, 175, 25),
                           limits=c(0.45, 1.10)) +
        scale_x_discrete(name = "Type of comments") +
        ggtitle("Boxplot of unique words per comments") +
        theme_bw()

char_comments = char_comments %>%
        reduce(rbind)
docs_freq = docs %>%
        map("comment_text") %>%
        #map(tolower) %>%
        #map(clean) %>%
        map(separate) %>%
        map(freqWord) %>%
        map2(names(docs),~mutate(.x, type=.y)) %>%
        reduce(rbind)             

docs_tf_idf = docs_freq %>%
        bind_tf_idf(word, type, Freq) %>%
        filter(tf_idf>0) %>%
        arrange(desc(tf_idf)) 

to_change = docs_tf_idf %>%
        inner_join(unknown, by=c("word", 'type'))

#correct the most one with the most important tf-idf value and we clean data

comments = readRDS("data/comments_preclean.rds")
comments_clean = comments %>%
        mutate(comment_text = clean(comment_text)) %>% 
        mutate(comment_text = removeWords(comment_text, c("supertr0ll", "tommy2010", "bunksteve",
                                                          "mothjer", "fan1967", "youcaltlas",
                                                          "notrhbysouthbanof", "bleachanhero",
                                                          "couriano", "jéské", stopwords("english")))) %>%
        mutate(comment_text = removeNumbers(comment_text))
#saveRDS(comments_clean, "comments_clean.rds")

comments_stem = comments_clean %>% 
        mutate(comment_text = stemDocument(comment_text, language = "english")) 
#saveRDS(comments_stem, "comments_stem.rds")

tf_idf_vocab = docs_tf_idf %>%
        mutate(word = removeNumbers(as.character(word))) %>% 
        #mutate(word = removeNumbers(as.character(clean(word)))) %>%
        #filter(length(word)>1) %>%
        #.[,"word"]
        .[1:15000,"word"]

#tf_idf_vocab = unlist(str_split(tf_idf_vocab, pattern =" "))[1:15000]

tf_idf_words = function(x){
        comment = strsplit(x, " ")
        comment = comment %>%
                map( function(y){
                        res = y [y %in% tf_idf_vocab #& nchar(y)>1
                                 ] 
                        res = paste(res, collapse = " ")
                }) 
        
        return(unlist(comment))
}


comments_tf_idf = comments %>%
        mutate(comment_text = tf_idf_words(removeNumbers(comment_text)))
#saveRDS(comments_tf_idf, "comments_tf_idf.rds")




set.seed(3456)
trainIndex = createDataPartition(comments$id_num, p = .8, list = FALSE)

commentsTrain = comments[ trainIndex,]

distribTrain = commentsTrain[,3:8] %>%
        map(sum) %>%
        reduce(rbind) %>%
        as.data.frame(row.names = FALSE)
distribTrain = rbind(distribTrain,nrow(commentsTrain)-sum(distribTrain)) %>%
        rename(count=V1) %>%
        mutate(percent = round(100*count/sum(count),2), sample="train")

distribTrain$type = c(names(commentsTrain[3:8]),"none")
distribTrain = distribTrain %>% arrange(desc(count))

commentsTest  = comments[-trainIndex,]

distribTest = commentsTest[,3:8] %>%
        map(sum) %>%
        reduce(rbind) %>%
        as.data.frame(row.names = FALSE)
distribTest = rbind(distribTest,nrow(commentsTest)-sum(distribTest)) %>%
        rename(count=V1) %>%
        mutate(percent = round(100*count/sum(count),2), sample="test")

distribTest$type = c(names(commentsTest[3:8]),"none") 

distrib = rbind(distribTrain, distribTest) %>%
        mutate(type=gsub("identity","id", type)) %>%
        mutate(type=gsub("severe","sev", type))

ggplot(distrib, aes(x = reorder(type, -percent), percent, fill = type)) +
        geom_col(show.legend = FALSE) +
        labs(x = NULL, y = "distribution") +
        geom_text(aes(label=paste0(percent,"%")), vjust=-0.3, color="black", size=3.5) +
        facet_wrap(~sample, ncol = 1, nrow=2, scales = "free") 


#keras

comments_stem = comments_tf_idf
comments_clean = comments


comments_train = comments[trainIndex,]
comments_tf_idf_train =comments_tf_idf[trainIndex,]
comments_clean_train = comments_clean[trainIndex,]
comments_stem_train = comments_stem[trainIndex,]
for(i in 1:4){
        if(i==1){final_data = comments_train}
        else if(i==2){final_data = comments_tf_idf_train}
        else if(i==3){final_data = comments_clean_train}
        else{final_data = comments_stem_train}
        
        #final_data = case_when(i,
        #        i==1~ comments,
        #        i==2~ comments_tf_idf,
        #        i==3~ comments_clean,
        #        i==4~ comments_stem
        #)
        
        
        
        max_words = 20000
        
        wordseq = text_tokenizer(num_words = max_words) %>%
                fit_text_tokenizer(final_data$comment_text)
        
        #word index
        word_index = wordseq$word_index
        maxl = 300
        
        # The 1 corresponds to the book methodology
        sequences = texts_to_sequences(wordseq, final_data$comment_text )
        train_data = pad_sequences(sequences, maxlen = maxl)
        
        train_label = as.matrix(final_data[,3:8])
        
        ndim = 50
        
        model = keras_model_sequential() %>%
                layer_embedding(input_dim = max_words, output_dim = ndim) %>%
                layer_gru(units = 64, return_sequences = TRUE) %>%
                #bidirectional(
                #        layer_lstm(units = 32)
                #) %>%
                layer_max_pooling_1d(  ) %>%
                layer_dropout(rate = 0.3) %>% 
                layer_gru(units = 64) %>%
                layer_dropout(rate = 0.3) %>% 
                layer_dense(units = 30, activation = "relu") %>%
                #layer_lstm(units = 32) %>%
                layer_dense(units = 6, activation = "sigmoid")
        
        model %>% compile(
                optimizer = "adam",
                loss = "binary_crossentropy",
                metrics = c("acc")
        )
        
        history = model %>% fit(
                train_data, train_label,
                epochs = 10,
                batch_size = 128,
                validation_split = 0.2,
                callbacks = list(
                        callback_model_checkpoint(paste0("model/checkpoints_model1_",i,".h5"), save_best_only = TRUE),
                        #callback_reduce_lr_on_plateau(monitor = "val_loss", factor = 0.1),
                        callback_early_stopping(monitor = "val_loss", min_delta = 0, patience = 0,
                                                verbose = 0, mode = c("auto", "min", "max"))
                        
                )
        )
        
        
        wgt100 = readRDS("data/glove.6B/glove_6B_100d.rds") %>%
                mutate(word=rm_white( gsub("[[:punct:]]"," ", word) ))
        wgt300 = readRDS("data/glove.6B/glove_6B_300d.rds") %>%
                mutate(word=rm_white( gsub("[[:punct:]]"," ", word) ))
        
        wordindex = unlist(wordseq$word_index)
        
        dic= data.frame(word=names(wordindex), key = wordindex,row.names = NULL) %>%
                arrange(key) %>% .[1:max_words,]
        
        for(k in 1:2){
                if(k==1){
                        wgt = wgt100
                }else{wgt = wgt300}
                
                
                w_embed = dic %>% 
                        left_join(wgt) 
                
                J = ncol(w_embed)
                
                w_embed = .[1:(max_words-1),3:J] %>%
                        mutate_all(as.numeric) %>%
                        mutate_all(funs(replace(., is.na(.), 0))) %>%
                        as.matrix()
                
                w_embed = rbind(rep(0, J), w_embed)
                
                matrix_w = mapply(w_embed, FUN=as.numeric)
                
                #ndim = J
                #vocab_size=nrow(dic)
                matrix_w =  matrix(data=matrix_w, ncol=J)
                #matrix_w = rbind(rep(0, 50), matrix_w)
                w_embed = list(array(matrix_w, c(max_words, J)))
                
                dim(matrix_w)
                
                #Example with embedding_weight
                model = keras_model_sequential() %>%
                        layer_embedding(input_dim = max_words, output_dim = ndim, input_length = maxl, weights = w_embed, trainable=FALSE) %>%
                        layer_gru(units = 64, return_sequences = TRUE) %>%
                        layer_max_pooling_1d(  ) %>%
                        layer_dropout(rate = 0.3) %>% 
                        layer_gru(units = 64) %>%
                        layer_dropout(rate = 0.3) %>% 
                        layer_dense(units = 30, activation = "relu") %>%
                        layer_dense(units = 6, activation = "sigmoid")
                
                get_layer(model, index = 1) %>%
                        set_weights(w_embed) %>%
                        freeze_weights()
                
                model %>% compile(
                        optimizer = "adam",
                        loss = "binary_crossentropy",
                        metrics = c("acc")
                )
                
                history = model %>% fit(
                        train_data, train_label,
                        epochs = 10,
                        batch_size = 128,
                        validation_split = 0.2,
                        callbacks = list(
                                callback_model_checkpoint(paste0("model/checkpoints_model",k,"_",i,".h5"), save_best_only = TRUE),
                                #callback_reduce_lr_on_plateau(monitor = "val_loss", factor = 0.1),
                                callback_early_stopping(monitor = "val_loss", min_delta = 0, patience = 0,
                                                        verbose = 0, mode = c("auto", "min", "max"))
                                
                                
                        )
                )
                
        }
        
        
} 
