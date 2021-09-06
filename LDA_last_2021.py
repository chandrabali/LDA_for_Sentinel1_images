from __future__ import print_function
import numba
import numpy as np
import glob
import struct
 
import os
#import gdal
import numpy as np
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
from skimage import io
import cv2
import pdb
font = ImageFont.truetype("arial.ttf", 8)
import pickle
 # Gensim imports########################################################################
import gensim
import gensim.corpora as corpora
from gensim.test.utils import datapath


output_path=""
colors=["maroon","darkblue","darkgreen","cyan","darkcyan","magenta","indigo","grey","peru","slateblue","mediumspringgreen","orangered"]
from numba import jit, njit

@jit(nopython=True) 
def distancefrom(cluster,vector):
    sq_sm=0
    for i in range(cluster.shape[0]):
        sq_sm=sq_sm + (cluster[i]-vector[i])**2
        
    return sq_sm
	
from numba import jit, njit
@jit(nopython=True, parallel=True)
def nearestNeighbour(clusters,vector):

    cluster_no=0
    minimum_dist=distancefrom(clusters[cluster_no],vector)
    
    for i in range(len(clusters)):
        current=clusters[i]
        #print(distancefrom(current,vector))
        if(distancefrom(current,vector) < minimum_dist):
            minimum_dist=distancefrom(current,vector)
            cluster_no=i
   
    return cluster_no


from numba import jit, njit
 
@jit(nopython=True, parallel=True)
def nearest_centroids(feat_mat_images,centroids):
    
    kmeanslabels_random=np.zeros(feat_mat_images.shape[0])
 
    for i in range(feat_mat_images.shape[0]):
        kmeanslabels_random[i]=nearestNeighbour(centroids,feat_mat_images[i])
     
    return kmeanslabels_random
        


def argmax_patch_topic(document_number,corpus,model): 
    
    doc_topics=model.get_document_topics(corpus[document_number])
     
    max_topic=doc_topics[0][0]  ## temporarily set the first topic as the maximum probable topic
    max_topic_prob=doc_topics[0][1]
    j=0
    while(j<len(doc_topics)):
                if(max_topic_prob<doc_topics[j][1]):
                    max_topic=doc_topics[j][0]
                    max_topic_prob=doc_topics[j][1]
                j=j+1  
         
    return max_topic


import pandas as pd
def All_topic_word_histogram(num_of_topics):
    
    topic_word=[] # a list of k topics, each row contains 10 lists (words and probabilities)

    for i in range(0,number_of_topics):
        temp_topic=[]
        for j in range(0,size_of_vocab):
            temp_word=list(lda_model.show_topic(i,size_of_vocab))[j]
            temp_topic.append(temp_word)
        topic_word.append(temp_topic)
        topic_term_prob_matrix=np.zeros((number_of_topics,size_of_vocab))
    
    for i in range(0,number_of_topics):
        for j in range(0,size_of_vocab):  
            for l in range(0,size_of_vocab):
                temp_word=topic_word[i][l][0]
                temp_prob=topic_word[i][l][1]
                if(j == int(math.floor(float(temp_word)))):
                   
                    topic_term_prob_matrix[i,j]=temp_prob
                    
    for i in range(0,num_of_topics):
        df_topic = pd.DataFrame(topic_term_prob_matrix[i,:]) #10 most important words in the given topic
        df_topic.plot.bar(figsize=(12,6))


def argmax_topic_doc(doc_no,corpus,lda_model):
    doc_max_topic=[]
    
    doc_topic_distr=lda_model.get_document_topics(corpus[doc_no])
    max_topic=doc_topic_distr[0][0]
    max_topic_prob=doc_topic_distr[0][1]
    
    for i in range(len(doc_topic_distr)):
        if(max_topic_prob<doc_topic_distr[i][1]):
                    max_topic=int(doc_topic_distr[i][0])
                    max_topic_prob=doc_topic_distr[i][1]
                
    return max_topic

#################### for a given document, find argmax_word_topic_dist 
################## function argmax_word_topic(document) considers integer id of each word assigned by gensim dictionary.not application to raw word tokens)

def argmax_word_topic(document,lda_model,corpus): 
    
    
    word_max_topic=[]
    max_topic="" 
    for w in range(0,len(document)):
        word_topic_dist= lda_model.get_term_topics(document[w])
        for i in range(0,len(word_topic_dist)): # for each word , finds the most probable topic
            max_=word_topic_dist[i][1]
           
            max_topic=word_topic_dist[i][0]
            j=0
            while(j<len(word_topic_dist)):
                if(max_<word_topic_dist[j][1]):
                    max_=word_topic_dist[j][1]
                    max_topic=word_topic_dist[j][0]
                j=j+1  
        word_max_topic.append(max_topic)
    return word_max_topic


def Draw_BoT_Model_Word(doc_no,data_arr,corpus,dictionary,lda_model,output_path):#i,data_arr,corpus,dictionary,lda_model
    
    
    ## All topics legend #############################
    img_topics=Image.new("RGB", (num_of_topics*60,100),(255,255,255))
    draw = ImageDraw.Draw(img_topics)
     
    xmin=20
    ymin=20
    xmax=60
    ymax=60
    for i in range(len(colors)):
        #text=str(i)
        draw.rectangle(((xmin, ymin), (xmax, ymax)), fill=colors[i])
    #draw.text((xmin, ymin),text,(255,255,255),font=font)
        xmin=xmax
        xmax+=60
    img_topics.save("Topics.jpg")
    
    BoT_vis_folder='BoT'
    if not(os.path.isdir(BoT_vis_folder)):
        os.mkdir(BoT_vis_folder)

    
    num_of_words=int(big_patch_dim/small_patch_dim)
    img = Image.new("RGB",(num_of_words*10,num_of_words*10),(255,255,255))
    
    draw = ImageDraw.Draw(img) # now a blank image to draw topics inside
    
    raw_doc= np.ndarray.tolist(data_arr[doc_no,:])
    converted_doc=[] #need to map real word tokens to their integer ids and generate the corresponding document
                 # a list of size N (each document has N words)
    
    for i in range(len(raw_doc)):
        
        converted_doc.append(dictionary.token2id[str(raw_doc[i])])
        
    topic_list=argmax_word_topic(converted_doc,lda_model,corpus)
    
    xmin=0
    ymin=0
    ymax=ymin+10
    for m in range(len(topic_list)):
        xmax=xmin+10
        topic=topic_list[m]
        text=str(topic)
        
        if (topic=="") :
             draw.rectangle(((xmin, ymin), (xmax, ymax)), fill=(0,0,0))
        else :           
            draw.rectangle(((xmin, ymin), (xmax, ymax)), fill=colors[int(topic)])
            draw.text((xmin, ymin),text,(255,255,255),font=font)
           
        xmin=xmax # for the next word
        
        if (xmax >= num_of_words*10): #next line after 64 words
            ymin=ymax   
            ymax=ymin+10
            xmin=0
     
    saveasname=BoT_vis_folder+"\\BoT_doc_"+str(doc_no)+".tif"
    #img.save(saveasname)
    return(img)
    #return(saveasname)
    

def draw_BoT_patch_based(corpus,model,doc_no,targetfolder):
    
    img = Image.open('BoT_blank.tif')
    draw = ImageDraw.Draw(img) # now a blank image to draw topics inside
    max_topic= argmax_topic_doc(doc_no,corpus,model)
    xmin=0
    ymin=0
    xmax=big_patch_dim
    ymax=big_patch_dim
    if (max_topic=="") :
             draw.rectangle(((xmin, ymin), (xmax, ymax)), fill=(0,0,0),outline="black")
    else :           
             draw.rectangle(((xmin, ymin), (xmax, ymax)), fill=colors[int(max_topic)],outline="black")
    
    saveasname= BoT_vis_folder+"\\BoT_doc_"+str(doc_no)+".tif"
    
    img.save(saveasname)
    



# =========================================================================================================================================
# Making the full bag of topics image 
# ===================================================================================================================================================
# In[ ]:

def draw_full_BoT(number_of_observations,data_arr,lda_model,corpus,dictionary): ## number of image documents and target forder for drawing the full BoT image
    
    import os
        
    Image.MAX_IMAGE_PIXELS=426406500
       
    im1=Image.new("RGBA", (x_size,y_size),(255,255,255))
    topiclist_corpus=[]
    for num in range(number_of_observations):
         topiclist_corpus.append([])
     
    for num in range(number_of_observations):
        i=0
        targetsize=big_patch_dim, big_patch_dim
        x_base=0
        y_base=0
        x_increment=big_patch_dim
        y_increment=big_patch_dim
        row_count=0
        col_count=0
        num_of_columns=int(x_size/big_patch_dim)
        num_of_rows=int(y_size/big_patch_dim)
        
        while(i<(num_of_rows * num_of_columns)): # e.g 100 colums,64 rows in the full images
            
            doc_no=i+num*(num_of_rows * num_of_columns)
             
            ######   task 1 : topic representation fpr the whole corpus ########
            raw_doc= np.ndarray.tolist(data_arr[doc_no,:])
            converted_doc=[] #need to map real word tokens to their integer ids and generate the corresponding document
                  # a list of size N (each document has N words)
            for ir in range(len(raw_doc)):
                converted_doc.append(dictionary.token2id[str(raw_doc[ir])])
            topiclist_corpus[num].append(argmax_word_topic(converted_doc,lda_model,corpus))
            
            
            ######  task 2: Draw the corpus ######################################
            #crop it  
            #im = Image.open(Draw_BoT_Model_Word(doc_no,data_arr,corpus,dictionary,lda_model,big_patch_dim,small_patch_dim,num_of_topics,output_path))
            #im = Draw_BoT_Model_Word(doc_no,data_arr,corpus,dictionary,lda_model,output_path)
            #im.thumbnail(targetsize)
            
            #im1.paste(im,(x_base,y_base))
            #col_count=col_count+1
            #x_base=x_base+x_increment
            #if(col_count==num_of_columns):
                #y_base=y_base+y_increment
                #x_base=0
                #col_count=0 ## set column count to zero for the next row
                #row_count=row_count+1 ## increment row count
            #i=i+1
        #im1.save("full_BoT"+str(num)+ ".tif")
    
    with open("topiclist_corpus.txt", "wb") as fp:   #Pickling
        pickle.dump(topiclist_corpus, fp)
   
 
        
def doc_topic_dist(lda_model, corpus, doc_no):

    import numpy as np
    import matplotlib.pyplot as plt
    example_list= lda_model.get_document_topics(corpus[doc_no])
    
    word = []
    frequency = []

    for i in range(len(example_list)):
        word.append('Topic ' + str(example_list[i][0]))
        frequency.append(example_list[i][1])

    indices = np.arange(len(example_list))
    plt.bar(indices, frequency, color='r')
    plt.xticks(indices, word, rotation='vertical')
    plt.tight_layout()
    plt.show()
 
def get_topic_per_doc(doc_no,data_arr,corpus,dictionary,lda_model):#i,data_arr,corpus,dictionary,lda_model
    
    # finds topic count distribution for a given image document
    
    raw_doc= np.ndarray.tolist(data_arr[doc_no,:])
    converted_doc=[] #need to map real word tokens to their integer ids and generate the corresponding document
    # a list of size N (each document has N words)
    for i in range(len(raw_doc)):
        
        converted_doc.append(dictionary.token2id[str(raw_doc[i])]) 
    topic_list=argmax_word_topic(converted_doc,lda_model,corpus)
    topic_count_list=[]
    for i in range(len(topic_list)):
        topic_count_list.append((topic_list[i],topic_list.count(topic_list[i])))
    topic_count_list=list(dict.fromkeys(topic_count_list)) # find number of occurences of each topic
    #with open("topic_count_each_image/topiccount_per_image.txt", "w") as output:
        #output.write(str(topic_count_list))  
    return(topic_count_list)#list of tuples containing topic-id and frequency

#%% find changes
def find_changes(data,x_size, y_size,big_patch_dim,small_patch_dim):
    change_val=131
    from itertools import combinations 
     
    # Get all permutations of [1, 2, 3] 
    perm = combinations(np.arange(len(data)), 2)
    #perm=[0,2]
    changes=[]
    number_of_changes=[]
    
    num_of_docs=int(x_size/big_patch_dim) * int(y_size/big_patch_dim)
    num_of_words=int(big_patch_dim/small_patch_dim) * int(big_patch_dim/small_patch_dim)
    # Print the obtained permutations 
    for p in perm: # corpus change combinations
        print("change between "+str(p[0]) + " - " +str(p[1])) 
        num_of_changed_words=0
        
        
        data1=data[p[0]] # corpus 1
        change_term=np.zeros((num_of_docs,num_of_words))# intially, we suppose no change, contain word by word changes between copuses
        data2=data[p[1]] #corpus 2
        #print("source"+ str(data1))
        #print("dest"+ str(data2))
        for i in range(len(data1)): # number of documents
            for j in range(len(data1[i])): # number of words
                ## handle null
                if (data1[i][j]=="" or data2[i][j]==""):
                    change_term[i][j]="-1"
                     
                #print("source: "+ str(data1[i][j]) + "dest: " + str(data2[i][j]) + " result: " + str(change_term[i][j]))
                elif(int(data1[i][j])==int(data2[i][j])):
                    change_term[i][j]=int(data1[i][j])
                    
                   # print("source: "+ str(data1[i][j]) + "dest: " + str(data2[i][j]) + "no change")
                else:
                    change_term[i][j]=change_val
                    num_of_changed_words+=1
        print(len(change_term))
        changes.append(change_term)
        number_of_changes.append(str(p)+ "- "+ str(num_of_changed_words) + " changes")
        
    with open("change.txt","w") as output:
        output.write(str(number_of_changes))  
    with open("change_data.txt", "wb") as fp:   #Pickling
        pickle.dump(changes, fp)   
        
    return changes

#%% check changes
def check_changes(data):
    for i in range(len(data)-1):
        print(data[i] == data[i+1])
#%% plot changes
def plot_changes(number_of_observations,data_arr,lda_model,corpus,dictionary,num_of_topics,big_patch_dim,small_patch_dim,x_size,y_size,output_path):
    
    import os
    change_vis_folder="change_in_docs"
    
    if not(os.path.isdir(change_vis_folder)):
        os.mkdir(change_vis_folder)
    
    with open("topiclist_corpus.txt", "rb") as fp:   # Unpickling
        topiclist_corpus = pickle.load(fp)
    #data=draw_full_BoT(number_of_observations,data_arr,lda_model,corpus,dictionary,num_of_topics,big_patch_dim,small_patch_dim,x_size,y_size,output_path)
    
      # number of observations
    change_maps= find_changes(topiclist_corpus,x_size, y_size,big_patch_dim,small_patch_dim) # changes between observations :corpus to corpus
    check_changes(change_maps)
    doc_no=0
    print("changes maps: " + str(len(change_maps)))
    num_of_rows=int(x_size/big_patch_dim) 
    num_of_columns=int(y_size/big_patch_dim) # num_of_rows * num_of_columns= M (number pf docs in corpus)
    num_of_words=int(big_patch_dim/small_patch_dim) # =N (number od words in the doc)
    doc_x_size=num_of_words*small_patch_dim *10# for drawing
    doc_y_size=num_of_words*small_patch_dim * 10
    bot_list=[]
    
    for num in range(len(change_maps)): 
        
        current_map=change_maps[num] # a corpus of M documents each having N words
           
        for i in range(num_of_rows*num_of_columns): # iterate through docs
            doc_img=Image.new("RGB",(doc_x_size,doc_y_size),(255,255,255))  
            draw = ImageDraw.Draw(doc_img)
    
            xmin=0
           
            ymin=0
            ymax=ymin + small_patch_dim * 10
            data_doc=current_map[i]
            #print("-----"+str(data_doc))
            for w in range(num_of_words * num_of_words): # iterate through words in doc
                
                topic=data_doc[w]
                
                xmax=xmin + small_patch_dim * 10 # 10 times zoomed
                if (topic==-1) :
                    text="-1"
                    draw.rectangle(((xmin, ymin), (xmax, ymax)), fill=(0,0,0))
                    draw.text((xmin, ymin),text,(255,255,255),font=font)## topic not found
                    
                elif(int(topic)==131): # a change shown in white
                    text="C"
                    draw.rectangle(((xmin, ymin), (xmax, ymax)), fill=(255,255,255))
                    draw.text((xmin, ymin),text,(0,0,0),font=font)# change is marked white
                else :  
                    text=str(topic)
                    draw.rectangle(((xmin, ymin), (xmax, ymax)), fill=colors[int(topic)]) 
                    draw.text((xmin, ymin),text,(0,0,0),font=font)
                xmin=xmax # for the next word
                 
        
                if (xmax >= num_of_words*small_patch_dim * 10): #next line after N words
                    ymin=ymax   
                    ymax=ymin+small_patch_dim * 10
                    xmin=0
     
            saveasname=change_vis_folder+"\\BoT_doc_"+str(doc_no)+".tif"
            #doc_img.save(saveasname)
            bot_list.append(doc_img)
            doc_no+=1
                
    for num in range(len(change_maps)):
        
        map_img=Image.new("RGB",(x_size,y_size),(255,255,255))
        i=0
        targetsize=big_patch_dim, big_patch_dim
        x_base=0
        y_base=0
        x_increment=big_patch_dim
        y_increment=big_patch_dim
        row_count=0
        col_count=0
        num_of_columns=int(x_size/big_patch_dim)
        num_of_rows=int(y_size/big_patch_dim)
        
        while(i<(num_of_rows * num_of_columns)): # e.g 100 colums,65 rows in the full images
            
            doc_no=i+num*(num_of_rows * num_of_columns)
            #im = Image.open(change_vis_folder+"\\BoT_doc_"+str(doc_no)+".tif")
            im=bot_list[doc_no]
            im.thumbnail(targetsize)
            
            map_img.paste(im,(x_base,y_base))
            col_count=col_count+1
            x_base=x_base+x_increment
            if(col_count==num_of_columns):
                y_base=y_base+y_increment
                x_base=0
                col_count=0 ## set column count to zero for the next row
                row_count=row_count+1 ## increment row count
            i=i+1
        map_img.save("full_ChangeMap"+str(num)+ ".tif")           
               
            
#%% SAR polarization methods

def hh_hv_avg_SAR(image_hh,image_hv):
	
    array_hh = image_hh
    array_hv = image_hv
    
    avg_array=np.zeros((array_hh.shape[0],array_hv.shape[1]))
    for i in range(array_hh.shape[0]):
        for j in range(array_hh.shape[1]):
            avg_array[i,j] = (int(array_hh[i,j]) + int(array_hv[i,j]))/2
            #print(array_hh[i,j], array_hv[i,j],avg_array[i,j])
    return avg_array

 
def hh_minus_hv_SAR(image_hh,image_hv):
	# hh-hv
    
    array_hh = image_hh
    array_hv = image_hv
    
    diff_array=np.zeros((array_hh.shape[0],array_hh.shape[1]))
    for i in range(array_hh.shape[0]):
         for j in range(array_hh.shape[1]):
            diff_array[i,j] = int(array_hh[i,j]) - int(array_hv[i,j])
             
    return diff_array
 

def combined_polarization(image_hh_list,image_hv_list):
    comb_arr_all=[]#np.zeros((1,16641,25673,3))
    for i in range(3):
        hh=image_hh_list[i][0:16641,0:25673]
        hv=image_hv_list[i][0:16641,0:25673]
        avg=hh_hv_avg_SAR(hh,hv)
        diff=hh_minus_hv_SAR(hh,hv)
        x_size=hh.shape[0]
        y_size=hh.shape[1]
        
        comb_arr=np.zeros((x_size,y_size,3))
    
        comb_arr[:,:,0]=hh
        print(np.amax(hh))
        comb_arr[:,:,1]=avg
        print(np.amax(avg))
        comb_arr[:,:,2]=diff
        print(np.amax(diff))
        #np.savetxt(str(i)+".txt",comb_arr,fmt="%d")
        comb_arr_all.append(comb_arr)
        
        
    return comb_arr_all


def read_image_SAR(Data,polarization):  ## fixed with combined polarization 
    import gdal
    import numpy as np
    
    Data_hh=Data+"hh\\"
    Data_hv=Data+"hv\\"
    img_list=[]
    image_hh_list=[]
    image_hv_list=[]
    #pdb.set_trace()
    for filename in glob.glob(Data_hh+"*.tiff"):
      
        print(filename)
        gtif = gdal.Open(filename)
        band=gtif.GetRasterBand(1)
        array_hh = band.ReadAsArray()
        print(np.amax(array_hh))
        print("hh" + str(array_hh.shape))
        image_hh_list.append(array_hh)
    
    for filename in glob.glob(Data_hv+"*.tiff"):
        print("reading file : " + filename)
        gtif = gdal.Open(filename)
        band=gtif.GetRasterBand(1)
        array_hv = band.ReadAsArray()
        print(np.amax(array_hv))
        print("hv" + str(array_hv.shape))
        
        image_hv_list.append(array_hv)

     ## adds all images
       
    return combined_polarization(image_hh_list,image_hv_list)
	
def read_image_MS(Data):   
    #import skimage
    from skimage import io
    img_list=[]
    for filename in glob.glob(Data+"*.jpg"):
        im=io.imread(filename)
		
        img_list.append(im)
    return img_list


def block1(Data,experiment):    
    bigger_patch_height=big_patch_dim
    bigger_patch_width=big_patch_dim
    small_patch_height= small_patch_dim # patches for creating words --each visual word is a small_patch_dim x small_patch_dim patch
    small_patch_width=small_patch_dim
    #############               ################            ################

	 
    img_full_list=read_image_SAR(Data,4)
		
    print("number of images:"+ str(len(img_full_list)))
    number_of_images=len(img_full_list)
    
    bigger_patches=[]
    for i in range(len(img_full_list)):
        img=img_full_list[i]
        s_image=img.shape 
        print("image shape: " + str(s_image))
        r=0
        c=0
        n=0	
        while ((r+ bigger_patch_height)<=s_image[0]):  # s_image[0] is height 
            while((c+bigger_patch_width)<=s_image[1]): #s_image[1] is width
				#index=str(r)+'+'+ str(r+bigger_patch_height)+"+"+ str(c) +'+'+str(c+bigger_patch_width)
                patch=img[r:r+bigger_patch_height,c:c+bigger_patch_width]
                n=n+1
				
                c=c+bigger_patch_width; 
                bigger_patches.append(patch)
				
            r=r+bigger_patch_height; #go to the next row and start with column 0 
            c=0; 		
		#print "number of bigger patches "+ str(len(bigger_patches))
    print ("number of bigger patches "+ str(len(bigger_patches)) )

            ################################## extract small patches (4 x 4) to create word ############################### 
    smaller_patch_list=[]
    for i in range(len(bigger_patches)): 
        img = bigger_patches[i]  
        s_image=img.shape 
        r=0
        c=0
        n=0 
        overlap_height= 0;
        overlap_width= 0;
    
        while ((r+ small_patch_height)<=s_image[0]):
            while((c+small_patch_width)<=s_image[1]):
                index=str(r)+'-'+ str(r+small_patch_height)+'-'+ str(c) +'-'+str(c+small_patch_width) #c is xmin , 'c+patch width is xmax'
                patch=img[r:r+small_patch_height,c:c+small_patch_width]
                n=n+1
                c=c+small_patch_width; 
                smaller_patch_list.append(patch)     
            r=r+small_patch_height; #go to the next row and start with column 0 
            c=0; 
    print ("number of small patches:" + str(len(smaller_patch_list)))
    #print "number of small patches:" + str(len(smaller_patch_list))
    
    feat_vec = np.zeros((small_patch_height * small_patch_width * no_of_channels), dtype=np.double)
    feat_mat=np.zeros((len(smaller_patch_list),(small_patch_height * small_patch_width * no_of_channels)))
    for i in range(len(smaller_patch_list)):
        feat_vec= np.resize(smaller_patch_list[i],(small_patch_height * small_patch_width * no_of_channels))
        feat_mat[i,:]=feat_vec
        ############################################# #     Block 3: k means clustering to create words  		#######################
    print ("shape of feature matrix: " + str(feat_mat.shape))
    
    centroids=np.loadtxt("C:/12months/centroids_19_18_labels_50words.txt")
    random_kmeanslabels=nearest_centroids(feat_mat,centroids)
    np.savetxt("random_kmeanslabels.txt",random_kmeanslabels,fmt="%d")
    print(" random kmeans labels saved")
    return(random_kmeanslabels)
    
import os   
def get_parentfolder(labeled_file,path):
        for root, dirs, files in os.walk(path):
            if labeled_file in files:
                return os.path.join(root, labeled_file)
            else:
                continue
    
def topic_dist_per_class(data_arr,corpus,dictionary,lda_model):
        
    label_topic_dist=[]
    for i in range(num_of_bigger_patches):
        label_topic_dist.append([])
    row_mul_col=(x_size/big_patch_dim)*(y_size/big_patch_dim)         
    #path="C:\\annotations"
    for i in range(num_of_bigger_patches):
        #month=int(i/row_mul_col) + 1
        #id_=(i%row_mul_col)+1
        #labeled_file=str(month)+"_"+str(id_)+".jpg"
        #if(get_parentfolder(labeled_file,path)is not None):
            #class_label=str(get_parentfolder(labeled_file,path)).split("/")[2]
        #else:
            #continue
        topic_counts=get_topic_per_doc(i,data_arr,corpus,dictionary,lda_model)
        label_topic_dist[i].append("BoT_doc_"+str(i))
        #label_topic_dist[i].append(str(class_label))
        label_topic_dist[i].append("_class_label_")
        label_topic_dist[i].append(str(topic_counts))
    
    filename="this_corpus_topicdist.txt"    
    with open(filename, "w") as output:
        output.write(str(label_topic_dist))  
 
def trainLDA(data_arr,experiment):
    
    
    #pdb.set_trace()
    #kmeanslabels=np.loadtxt("C:\\Users\\karm_ch\\Documents\\code\\LDA_revised\\6months12topics20wordscombined_pol\\kmeans_6months12topics20words.txt")
    
    print ('shape of  data array'+ str(data_arr.shape))
     
    word_count_documents=np.zeros((num_of_bigger_patches,size_of_vocab)) # contains word distribution per image document
    
    for d in range(0,num_of_bigger_patches):   
        
        for i in range(0,num_of_words):
            check_val=data_arr[d][i]
            for n in range(0,size_of_vocab):
                if int(check_val)==int(n):
                   
                    word_count_documents[d][n]=int(word_count_documents[d][n])+1
   
    text=[]
    for i in range(data_arr.shape[0]): # for each image
        document=[]
        for j in range(data_arr.shape[1]):
            document.append(str(data_arr[i,j])) # add each word
        text.append(list(document))
    

    corpus=[]
    for i in range(word_count_documents.shape[0]):
        document=[]
        for j in range(word_count_documents.shape[1]):
            document.append([j,int(word_count_documents[i,j])])
        corpus.append(tuple(document))
    
    with open("corpus.txt", "wb") as fp:   #Pickling
        pickle.dump(corpus, fp)
    diction = corpora.Dictionary(text)
    diction.save("polar_dictionary_lda.pkl")
    
    dictionary = corpora.Dictionary.load("polar_dictionary_lda.pkl")
   
    
    # Build LDA model
    lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                               id2word=diction,
                                               num_topics=num_of_topics)
    
    #save the model
    temp_file = datapath("model")
    lda_model.save(temp_file)
    import joblib

    joblib.dump(lda_model, experiment +'.jl')   
    return temp_file,dictionary
   
#pdb.set_trace()
def make_data_array(data_folders,experiment,pol_text):
    import numpy as np
    
    data_arr_full=[]
    corpus=[]
    #number_of_observations=len(data_folders)
    global num_of_bigger_patches
    #kmeanslabels=np.loadtxt("C:/users/karm_ch/documents/code/2018_1_3_months_randomDict/random_kmeanslabels.txt")#
    kmeanslabels=block1(data_folders,experiment)
    num_of_bigger_patches=int(kmeanslabels.shape[0]/num_of_words)
    print("function:make data arr - number of bigger patches"+str(num_of_bigger_patches))    
    data_arr=np.resize(kmeanslabels,(num_of_bigger_patches,num_of_words))
    
    return data_arr 

def make_corpus(data_arr): # creates the bag if words representation
     word_count_documents=np.zeros(( num_of_bigger_patches,size_of_vocab)) # contains word distribution per image document
    
     for d in range(0,num_of_bigger_patches):   
         for i in range(0,num_of_words):
            check_val=data_arr[d][i]
            for n in range(0,size_of_vocab):
                if int(check_val)==int(n):
                   word_count_documents[d][n]=int(word_count_documents[d][n])+1                  
     text=[]
     for i in range(data_arr.shape[0]): # for each image
        document=[]
        for j in range(data_arr.shape[1]):
            document.append(str(data_arr[i,j])) # add each word
        text.append(list(document))
         
     corpus=[]
     for i in range(word_count_documents.shape[0]):
         document=[]
         for j in range(word_count_documents.shape[1]):
             document.append([j,int(word_count_documents[i,j])])
         corpus.append(tuple(document))
        
     return corpus
    
 
def main_():
    
    # Gensim imports########################################################################
    import gensim
    import gensim.corpora as corpora
    from gensim.test.utils import datapath
    import datetime
    import time
    import numpy as np
    print(datetime.datetime.now())
    
    ## globals
    global size_of_vocab, sentinel,polarization,big_patch_dim,small_patch_dim,num_of_words,lda_model, num_of_bigger_patches
    global year, x_size,y_size,no_of_channels,pol_text,num_of_topics,x_part_global, number_of_observations
    
    start_time=time.time()
    ## initializations
    size_of_vocab = 50#int(input("Please enter the size of vocab: "))
    
    sentinel= 1#int(input("For sentinel1 images, please enter 1, for sentinel2 images enter 2: "))
    num_of_topics= 12#int(input("Please enter the number of topics: "))
    big_patch_dim=256#int(input("please enter the height or width(both same) of the bigger patches :"))
    small_patch_dim=4#int(input("please enter the height or width(both same) of the smaller patch :"))
    num_of_words=int(big_patch_dim/small_patch_dim)* int(big_patch_dim/small_patch_dim)
    x_size=25673#int(input("please enter the width of the images(x direction) : "))
    y_size=16637#int(input("please enter the height of the images(y direction) : "))
    number_of_observations=4 # number of months
    no_of_channels=3
    #experiment=  input("Please enter a name for the experiment and remember to change the data path in line 505: ")
	
    if(sentinel==2):
        no_of_channels=int(input("Enter the number of channels/bands (default is 1) :"))
        polarization=0
    
    if(sentinel==1):
        polarization = 4#int(input("Choose polarization method: (Enter 1 for only hh or 2 for avg(hh+hv) or 3 for hh-hv and 4 for all combined : "))
    
    pol_text="" 
    if(sentinel==1 and polarization ==1):
        pol_text="HH"
    elif(sentinel==1 and polarization==2):
        pol_text="avg(HH,HV)"
    elif(sentinel==1 and polarization==3):
        pol_text="HH-HV"
    elif(sentinel==1 and polarization==4):
        pol_text="combined_pol"
        no_of_channels=3
    
    year=2018
     
     
    
    data_folders="C:\\12months\\"
     
    experiment="24_mon_4_7_randomDict"
    if not(os.path.isdir(experiment)):
        os.mkdir(experiment)
    os.chdir(experiment)
	
    #data_arr=make_data_array(data_folders, experiment,pol_text)
    kmlabels= np.load("C:\\articles\\LDA_change_detection\\result_backup\\kmlabels\\1_kmeanslabels_4_to_7_randomDict_50w.npy")
    data_arr=np.resize(kmlabels, (6400 * 4,4096))
    num_of_bigger_patches=data_arr.shape[0]
    corpus=make_corpus(data_arr)
    
    
    #lda_model_path,dictionary=trainLDA(data_arr,experiment)
    import joblib
    #lda_model = gensim.models.ldamodel.LdaModel.load(lda_model_path)
    lda_model=joblib.load("C:/articles/LDA_change_detection/deliverables_to_ExtremeEarth/24_mon_lda_model_v_21.jl") 
	
    dictionary = corpora.Dictionary.load("C:/articles/LDA_change_detection/deliverables_to_ExtremeEarth/polar_dictionary_lda.pkl")  
    lda_model.update(corpus)
    joblib.dump(lda_model,"24_mon_lda_model_v_21_updated.jl")
    draw_full_BoT(number_of_observations,data_arr,lda_model,corpus,dictionary)
    #topic_dist_per_class(data_arr,corpus,dictionary,lda_model)
    
    #plot_changes(number_of_observations,data_arr,lda_model,corpus,dictionary,num_of_topics,big_patch_dim,small_patch_dim,x_size,y_size,output_path)
    
    os.chdir("..") 
    print("---the algorithm took  %s seconds ---" % (time.time() - start_time)) 
main_()