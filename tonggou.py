#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 10:05:51 2017

@author: gaoyiwei
"""

#ave ave ave
#max max max   is wrong   ave比max好很多

import tensorflow as tf

class cnnattmodel(object):
    def __init__(self,config,is_training,embedding):       
        
        #canshu
        self.wordfiltersize=config.wordfiltersize
        self.senfiltersize=config.senfiltersize
        
        self.wordEmbeddingSize=config.wordEmbeddingSize
        
        self.sentencesnum=config.sentencesnum
        self.wordsnum=config.wordsnum
        
        self.batchsize=config.batchsize
        self.classnum=config.classnum
        
        self.wordtosen_hidsize=config.wordtosen_hidsize
        self.sentotext_hidsize=config.sentotext_hidsize
        
        self.wordase=config.wordase
        self.senase=config.senase
        self.ngram=config.ngram
        self.keep_prob=config.keep_prob
        
        #add placeholder
        self.x=tf.placeholder(tf.int32,[None,self.sentencesnum,self.wordsnum])
        self.y=tf.placeholder(tf.int64,[None])
        
        #embedding lookup
        self.inpute=tf.nn.embedding_lookup(embedding,self.x) #batch,sens,words,wordsize
        self.inpute1=tf.reshape(tf.concat(self.inpute,0),[-1,self.wordsnum,self.wordEmbeddingSize])  #0 is axis  batch*sens,words,emds
        self.inpute_expand=tf.expand_dims(self.inpute1,-1)#batch*sens,words,embsize,1
        
        self.initial_weight()
        
        #cnn
        word_cx=tf.nn.conv2d(self.inpute_expand,self.word_filterr,strides=[1,1,1,1],padding='VALID')#batch,height-filter+1,1,channel
        wbias=tf.nn.bias_add(word_cx,self.word_bias)
        wrecx=tf.nn.relu(wbias) #batch,words-filter+1,1,channel  6,2,1,3
        
        #改为max-pooling
        maxwrecx=tf.nn.avg_pool(wrecx,ksize=(1,self.wordsnum-self.ngram+1,1,1),strides=[1,1,1,1],padding='VALID')#[batch_size, 1, 1, num_filters]
        h_pool_flat=tf.reshape(maxwrecx,[-1,self.wordtosen_hidsize]) #shape should be:[None,hid]. here this operation has some result as tf.sequeeze().e.g. x's shape:[3,3];tf.reshape(-1,x) & (3, 3)---->(1,9)
        with tf.name_scope("dropout"):           
            wsenp=tf.nn.dropout(h_pool_flat,keep_prob=self.keep_prob) #[None,num_filters_total]
        
        '''
        #attention 是否需要attention呢，或者是max-pooling
        wcx1=tf.reshape(wrecx,[-1,self.wordsnum-self.wordfiltersize+1,self.wordtosen_hidsize]) #batch*sens,words-filters+1,channel
        wcx2=tf.reshape(wrecx,[-1,self.wordtosen_hidsize])#batch*sens*words-filters+1,channel
        wa=tf.matmul(wcx2,self.ww_word)+self.wb_word
        wcx3=tf.nn.tanh(wa)
        wh_r=tf.reshape(wcx3,[-1,self.wordsnum-self.wordfiltersize+1,self.wordase]) #batch*sens,words-filter+1,ase
        wsimi=tf.multiply(wh_r,self.context_word)  #batch*sens,words-filter+1,ase
        wato=tf.reduce_sum(wsimi,2) #batch*sens,words-filter+1
        wmaxhang=tf.reduce_max(wato,1,True) #batch*sens,1
        wpattention=tf.nn.softmax(wato-wmaxhang)#batch*sens,words-filter+1
        wpae=tf.expand_dims(wpattention,2)#batch*sens,words-filter+1,1
        wsenp=tf.multiply(wpae,wcx1) #batch,words-filter+1,channel
        wsenp=tf.reduce_sum(wsenp,1) #batch*sens,wordtosen_hid   
        '''
        
        '''
        #cnn sententce aspect
        sinput=tf.reshape(wsenp,[-1,self.sentencesnum,self.wordtosen_hidsize])#batch,sens,hid
        sinput_expand=tf.expand_dims(sinput,-1)#batch,sens,hid,1
        #cnn
        scx=tf.nn.conv2d(sinput_expand,self.sentence_filter,strides=[1,1,1,1],padding='VALID')#batch,height-filter+1,1,channel
        sbias=tf.nn.bias_add(scx,self.sentence_bias)
        srecx=tf.nn.relu(sbias) #batch,words-filter+1,1,channel  3,2,1,3
        #attention
        scx1=tf.reshape(srecx,[-1,self.sentencesnum-self.senfiltersize+1,self.sentotext_hidsize]) #batch,words-filters+1,filters
        scx2=tf.reshape(srecx,[-1,self.sentotext_hidsize])#batch*words-filters+1,filters
        sa=tf.matmul(scx2,self.ww_sentence)+self.wb_sentence
        scx3=tf.nn.tanh(sa)
        sh_r=tf.reshape(scx3,[-1,self.sentencesnum-self.senfiltersize+1,self.senase]) #batch,words-filter+1,ase
        ssimi=tf.multiply(sh_r,self.context_sentence)  #batch,words-filter+1,ase
        sato=tf.reduce_sum(ssimi,2) #batch,words-filter+1
        smaxhang=tf.reduce_max(sato,1,True) #batch,1
        spattention=tf.nn.softmax(sato-smaxhang)#batch,words-filter+1
        spae=tf.expand_dims(spattention,2)#batch,words-filter+1,1
        ssenp=tf.multiply(spae,scx1) #batch,words-filter+1,filters
        ssenp=tf.reduce_sum(ssenp,1) #batch,sentotext_hid
        '''
        '''
        #lstm sentence 
        cnnsinput=tf.reshape(wsenp,[-1,self.sentencesnum,self.wordtosen_hidsize])#batch,sens,hid
        cnnrsenl=tf.split(cnnsinput,self.sentencesnum,1) #list,sens,every is batch*1*hid
        cnnrsinput3=[tf.squeeze(x,[1]) for x in cnnrsenl]
        cnnrslstm_cell=tf.nn.rnn_cell.BasicLSTMCell(self.sentotext_hidsize)
        if self.keep_prob<1:
            cnnrslstm_cell=tf.nn.rnn_cell.DropoutWrapper(cnnrslstm_cell,output_keep_prob=self.keep_prob)
        cnnrscell=tf.nn.rnn_cell.MultiRNNCell([cnnrslstm_cell]*1)       
        self.cnnsinitial_state=cnnrscell.zero_state(self.batchsize,tf.float32)  #initial state  t=0 :c and h                   
        cnnrsoutput=[]
        cnnstate=self.cnnsinitial_state   
        with tf.variable_scope("cnnLSTM_layers"):          
            for time_step,data in enumerate(cnnrsinput3):
                if time_step>0:
                    tf.get_variable_scope().reuse_variables()
                cell_output,cnnstate=cnnrscell(data,cnnstate)
                cnnrsoutput.append(cell_output)
        
        #average pooling
        cnnrhidden_sen_1=tf.stack(cnnrsoutput,axis=1) #batch,sens,hidden
        ssdocp=tf.reduce_mean(cnnrhidden_sen_1,1) #batch hidden
        '''
        
        '''
        #attention  or  max--pooling or average pooling
        cnnrhidden_sen_1=tf.stack(cnnrsoutput,axis=1) #batch,sens,hidden
        cnnrhidden_sen_2=tf.reshape(cnnrhidden_sen_1,[-1,self.sentotext_hidsize]) #batch*sen,hid       
        cnnrsa=tf.matmul(cnnrhidden_sen_2,self.ww_sentence)+self.wb_sentence
        cnnrsh_r1=tf.nn.tanh(cnnrsa)
        cnnrsh_r=tf.reshape(cnnrsh_r1,[-1,self.sentencesnum,self.sentotext_hidsize])#batch,sens,hid
        cnnrssimi=tf.multiply(cnnrsh_r,self.context_sentence)
        cnnrsato=tf.reduce_sum(cnnrssimi,2)  #batch,sens
        cnnrsmaxhang=tf.reduce_max(cnnrsato,1,True)
        cnnrsatt=tf.nn.softmax(cnnrsato-cnnrsmaxhang) #batch,sensattention
        cnnrsae=tf.expand_dims(cnnrsatt,2) #batch,sens,1
        ssdocp=tf.multiply(cnnrsae,cnnrhidden_sen_1) #batch,sens,hid
        ssdocp=tf.reduce_sum(ssdocp,1) 
        '''
        #print ('end cnn')
        #end cnn
        
#---------------------------------------------------------------------------------        
        #this code block is userd for rnn-gram
        self.rinpute=tf.nn.embedding_lookup(embedding,self.x)  #batch*sens*words*embedding
        self.rinpute1=tf.reshape(tf.concat(self.rinpute,0),[-1,self.wordsnum,self.wordEmbeddingSize])  #0 is axis  batch*sens,words,emds
        self.rinputsnew=[] #batch*sens,words,emb   gram=2
        for i in range(self.batchsize*self.sentencesnum):
            print (i)
            doc=self.rinpute1[i,:,:]
            docnew=[]
            for j in range(self.wordsnum-self.ngram+1):  #wordslength-gram+1
                #print (j)
                gram=doc[j,:]
                for g in range(self.ngram-1):
                    gram=gram+doc[j+1+g,:]
                docnew.append(gram/self.ngram)
            self.rinputsnew.append(docnew)  #batch*sens,words-gram+1,emb
        #print ('end rnn ngram')
        self.i1=tf.reshape(tf.concat(self.rinputsnew,0),[-1,self.sentencesnum,self.wordsnum-self.ngram+1,self.wordEmbeddingSize])
        self.rinput1=tf.reshape(tf.concat(self.i1,0),[-1,self.wordsnum-self.ngram+1,self.wordEmbeddingSize]) #batch*sens,words,embedding
        
        #lstm层
        rinput2=tf.split(self.rinput1,self.wordsnum-self.ngram+1,1)  #list,words,batch*sens,1,emb       
        self.rinput3=[tf.squeeze(x,[1]) for x in rinput2] #list,lenth is words,every is batch*sens,emb        
        lstm_cell=tf.nn.rnn_cell.BasicLSTMCell(self.wordtosen_hidsize)
        if self.keep_prob<1:
            lstm_cell=tf.nn.rnn_cell.DropoutWrapper(lstm_cell,output_keep_prob=self.keep_prob)
        cell=tf.nn.rnn_cell.MultiRNNCell([lstm_cell]*1)       
        self.winitial_state=cell.zero_state(self.batchsize*self.sentencesnum,tf.float32)  #initial state  t=0 :c and h                   
        self.rwoutput=[]
        state=self.winitial_state   
        with tf.variable_scope("LSTM_layerw"):
            for time_step,data in enumerate(self.rinput3):
                if time_step>0:
                    tf.get_variable_scope().reuse_variables()
                cell_output,state=cell(data,state)
                self.rwoutput.append(cell_output)
        
        #max-pooling
        rhidden_state_1=tf.stack(self.rwoutput,axis=1) #batch*sens,words,wordtosen_hidsize
        rsenp=tf.reduce_mean(rhidden_state_1,1)#batch*sens,hid
        
        #tonggou
        concatesenp=tf.concat([wsenp,rsenp],1) #batch*sens,2*hid
        '''
        #attention
        rhidden_state_1=tf.stack(self.rwoutput,axis=1) #batch*sens,words,wordtosen_hidsize
        rhidden_state_2=tf.reshape(rhidden_state_1,[-1,self.wordtosen_hidsize]) #batch*sens*words,hid
        ra=tf.matmul(rhidden_state_2,self.rww_word1)+self.rwb_word1
        rh_r1=tf.nn.tanh(ra)
        rh_r=tf.reshape(rh_r1,[-1,self.wordsnum-self.ngram+1,self.wordtosen_hidsize]) #batch*sens,words,hid
        rsimi=tf.multiply(rh_r,self.rcontext_word1)  #batch*sens,words,hid
        rato=tf.reduce_sum(rsimi,2) #batch*sens,words
        rmaxhang=tf.reduce_max(rato,1,True) #batch*sens,1
        rpattention=tf.nn.softmax(rato-rmaxhang)#batch*sens,words
        rpae=tf.expand_dims(rpattention,2)#batch*sens,words,1
        rsenp=tf.multiply(rpae,rhidden_state_1)
        rsenp=tf.reduce_sum(rsenp,1) #batch*sens,wordtosen_hid
        '''
        
        #sentence level
        rinputss=tf.reshape(concatesenp,[-1,self.sentencesnum,self.wordtosen_hidsize*2])#bathc,sens,hid
        
        #lstm
        rsenl=tf.split(rinputss,self.sentencesnum,1) #list,sens,every is batch*1*hid
        rsinput3=[tf.squeeze(x,[1]) for x in rsenl]
        rslstm_cell=tf.nn.rnn_cell.BasicLSTMCell(self.sentotext_hidsize)
        if self.keep_prob<1:
            rslstm_cell=tf.nn.rnn_cell.DropoutWrapper(rslstm_cell,output_keep_prob=self.keep_prob)
        rscell=tf.nn.rnn_cell.MultiRNNCell([rslstm_cell]*1)       
        self.sinitial_state=rscell.zero_state(self.batchsize,tf.float32)  #initial state  t=0 :c and h                   
        rsoutput=[]
        state=self.sinitial_state   
        with tf.variable_scope("LSTM_layers"):          
            for time_step,data in enumerate(rsinput3):
                if time_step>0:
                    tf.get_variable_scope().reuse_variables()
                cell_output,state=rscell(data,state)
                rsoutput.append(cell_output)
                
        #average pooling
        rhidden_sen_1=tf.stack(rsoutput,axis=1) #batch,sens,hidden
        self.rdocp=tf.reduce_mean(rhidden_sen_1,1) #batch,hid
        '''
        #attention
        rhidden_sen_1=tf.stack(rsoutput,axis=1) #batch,sens,hidden
        rhidden_sen_2=tf.reshape(rhidden_sen_1,[-1,self.sentotext_hidsize]) #batch*sen,hid       
        rsa=tf.matmul(rhidden_sen_2,self.rww_sentence1)+self.rwb_sentence1
        rsh_r1=tf.nn.tanh(rsa)
        rsh_r=tf.reshape(rsh_r1,[-1,self.sentencesnum,self.sentotext_hidsize])#batch,sens,hid
        rssimi=tf.multiply(rsh_r,self.rcontext_sentence1)
        rsato=tf.reduce_sum(rssimi,2)  #batch,sens
        rsmaxhang=tf.reduce_max(rsato,1,True)
        rsatt=tf.nn.softmax(rsato-rsmaxhang) #batch,sensattention
        rsae=tf.expand_dims(rsatt,2) #batch,sens,1
        rdocp=tf.multiply(rsae,rhidden_sen_1) #batch,sens,hid
        rdocp=tf.reduce_sum(rdocp,1) 
        '''                       
        #end rnn
#------------------------------------------------------------     
        #concat rnn and cnn
        #self.final_docp=tf.concat([ssdocp,rdocp],1)
        #final mlp
        self.logits=tf.matmul(self.rdocp,self.w1)+self.b1 #bath,classe
        #define loss jiaochashang
        with tf.name_scope("losscost_layer"):
            self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y,logits=self.logits)
            self.cost = tf.reduce_mean(self.loss)
            
        #define accuracy
        with tf.name_scope("accuracy"):
            self.prediction = tf.argmax(self.logits,1)
            correct_prediction = tf.equal(self.prediction,self.y)
            self.correct_num=tf.reduce_sum(tf.cast(correct_prediction,tf.float32))
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32),name="accuracy")
        if not is_training:
            return
        
        #self.cost=0
        #you hua
        self.globle_step = tf.Variable(0,name="globle_step",trainable=False)
        self.lr = tf.Variable(0.0,trainable=False)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars),config.max_grad_norm)     
        optimizer = tf.train.GradientDescentOptimizer(self.lr)
        #optimizer.apply_gradients(zip(grads, tvars))
        self.train_op=optimizer.apply_gradients(zip(grads, tvars))

        self.new_lr = tf.placeholder(tf.float32,shape=[],name="new_learning_rate")
        self._lr_update = tf.assign(self.lr,self.new_lr)
        
    def assign_new_lr(self,session,lr_value):
        session.run(self._lr_update,feed_dict={self.new_lr:lr_value}) 
    
    def initial_weight(self):        
         with tf.name_scope("cnn-attention"):
             self.word_filterr=tf.get_variable("word_filter",[self.wordfiltersize,self.wordEmbeddingSize,1,self.wordtosen_hidsize])
             self.word_bias=tf.get_variable("word_bias",[self.wordtosen_hidsize])                        
             #self.ww_word=tf.get_variable("ww_word",shape=[self.wordtosen_hidsize,self.wordase])
             #self.wb_word=tf.get_variable("wb_word",shape=[self.wordase])
                   
             #self.sentence_filter=tf.get_variable("sen_filter",[self.senfiltersize,self.wordtosen_hidsize,1,self.sentotext_hidsize])
             #self.sentence_bias=tf.get_variable("sen_bias",[self.sentotext_hidsize])
             #self.ww_sentence=tf.get_variable("ww_sen",shape=[self.sentotext_hidsize,self.senase])
             #self.wb_sentence=tf.get_variable("wb_sen",shape=[self.senase])
            
             #self.context_word=tf.get_variable("context_word",shape=[self.wordase])
             #self.context_sentence=tf.get_variable("context_sen",shape=[self.senase])
         
         #with tf.name_scope("rnn-attention"):            
            #self.rww_word1=tf.get_variable("rww_word22",shape=[self.wordtosen_hidsize,self.wordtosen_hidsize])
            #self.rwb_word1=tf.get_variable("rwb_word22",shape=[self.wordtosen_hidsize])
        
            #self.rww_sentence1=tf.get_variable("rww_sen22",shape=[self.sentotext_hidsize,self.sentotext_hidsize])
            #self.rwb_sentence1=tf.get_variable("rwb_sen22",shape=[self.sentotext_hidsize])
            
            #self.rcontext_word1=tf.get_variable("rcontext_word22",shape=[self.wordtosen_hidsize])
            #self.rcontext_sentence1=tf.get_variable("rcontext_sen22",shape=[self.sentotext_hidsize]) 
            
            
         with tf.name_scope("final"):
             self.w1=tf.get_variable("w22",shape=[self.sentotext_hidsize,self.classnum])
             self.b1=tf.get_variable("b22",shape=[self.classnum])   
        
                