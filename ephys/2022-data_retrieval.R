####### new R script to retrive data

####### data loading 
cbPalette<-c('#4758A6','#BC0404','#A8ABD5','#DD9E89')

#########################################################################################################
####### Getting the vivarium file
#########################################################################################################
	if(Sys.info()['sysname']=='Windows'){#
	        dir<-"Y:/Vaissiere/ARCHIVE/VIVARIUM";for(i in dir){#
	                setwd(i)#
	                setwd(i)#
	        }#
	}else{#
	        dir<-'/Volumes/RumbaughMillerData/VIVARIUM';for(i in dir){#
	                setwd(i)#
	                setwd(i)#
	        }#
	}#

	viv<-data.table(read.xlsx('Vivarium (Autosaved1).xlsx',sheetIndex = 8)); viv1<-viv#
	viv<-viv[Genotype == "Thy1-ChR2-YFP+/-",c(3,4,5,15,16,17,22)]#
	viv<-rename(viv, c(Pedigree.. = "sID", Transnetyx = "WellPlate"))#
	viv$sID<-as.numeric(as.character(viv$sID))#

	files<-list.files(dir, pattern = '.csv')#
	out<-NULL#
	for (i in 1:length(files)){#
	  data<-data.table(read.csv(files[i]))#
	  out<-rbind(out, data, fill = TRUE)#
	}#
	out<-out[,c(1,4,6,7)]#
	out$Neomycin<-as.character(out$Neomycin)#
	out$Neomycin<-gsub(' ','',out$Neomycin)#
	out$geno[out$Neomycin == "+"]<-"het"#
	out$geno[out$Neomycin == "-"]<-"wt"#
	out<-rename(out, c(Sample = 'sID'))
	#
	str(out)#
	str(viv)#
	#
	viv.final<-merge(viv, out, by=c("sID","WellPlate"))
	viv.final

#########################################################################################################
####### Ref file
#########################################################################################################
	# LOAD - REF FILE ####
	if(Sys.info()['sysname']=='Windows'){
	dir<-'Y:/Vaissiere/ARCHIVE/00-ephys temp/';for(i in dir){
	        setwd(i)
	        setwd(i)
	}
	}else{
	dir<-'/Volumes/MillerRumbaughLab/Vaissiere/00-ephys temp';for(i in dir){
	        setwd(i)
	        setwd(i)
	}
	}

	# find all the matlab file within the ephys temp
	files<-list.files(pattern = '.mat', recursive = TRUE)
	files<-files[-grep(".pdf", files)]


	# look for the correpsonpding table with experiment description
	exp.d<-read.csv('Description.csv')
	exp.d$files<-paste("MATLA/",
	      exp.d$month,"/",
	      formatC(exp.d$day, width = 2, flag="0"),"/",
	      "experiment", formatC(exp.d$exp, width = 3, flag="0"), "trial",formatC(exp.d$trial, width = 3, flag="0"),".mat",
	      sep = "")
	tail(exp.d$group)

#########################################################################################################
####### PIEZO DATA
#########################################################################################################