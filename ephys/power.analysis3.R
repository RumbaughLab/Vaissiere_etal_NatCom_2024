library(plyr)
library(ggplot2)
library(data.table)
dir<-'Y:/Vaissiere/00-ephys temp/';for(i in dir){
  setwd(i)
  setwd(i)
}

dir<-'/Volumes/MillerRumbaughLab/Vaissiere/00-ephys temp/';for(i in dir){
        setwd(i)
        setwd(i)
}


toadd1<-read.csv('power.setAS.csv', colClasses = c(rep("numeric",5),"POSIXct"), header = TRUE)

wave.length<-470
mat.lab.input<-c(0,0.1,0.5,1,2,3,4,5)
measured.value<-c(0.088,0.1,0.292,0.525,0.952,1.337,1.69,2.02)
power.dat<-data.table(mat.lab.input,measured.value,wave.length)
mod<-lm(measured.value ~ mat.lab.input, power.dat); summary(mod)
power.dat$intercept<-coef(mod)[1]
power.dat$slope<-coef(mod)[2]
power.dat$date.time<-Sys.time()

power.dat<-rbind(power.dat, toadd1)
write.csv(power.dat, "power.setAS.csv", row.names = FALSE)

wave.length<-625
mat.lab.input<-c(0,0.1,0.5,1,2,3,4,5)
measured.value<-c(0.068, 0.095, 0.269, 0.472, 0.885, 1.28, 1.66, 2.01)
power.dat<-data.table(mat.lab.input,measured.value,wave.length)
mod<-lm(measured.value ~ mat.lab.input, power.dat); summary(mod)
power.dat$intercept<-coef(mod)[1]
power.dat$slope<-coef(mod)[2]
power.dat$date.time<-Sys.time()

power.dat<-rbind(power.dat, toadd1)
write.csv(power.dat, "power.setAS.csv", row.names = FALSE)


windows(); quartz()
ggplot(power.dat, aes(x=mat.lab.input, y=measured.value, group = 1)) +
        geom_smooth(se = TRUE, method = "lm", color="black")+
        geom_point()

toadd1$wave.length<-as.factor(toadd1$wave.length)
ggplot(toadd1, aes(x=mat.lab.input, y=measured.value, group = wave.length)) +
        #geom_smooth(aes(color=wave.length), se = TRUE, method = "lm")+
        geom_point(aes(color=wave.length))



# new power set ####
pow.set<-data.table(desired.mWmm2=c(0,2.5,5,7.5,10,12.5,15,17.5,20,30,40,50,60))
pow.set<-ddply(pow.set,
               .(desired.mWmm2),
               mutate,
               mW.equi = 0.1*0.1*pi*desired.mWmm2,
               toinput.inmat = mW.equi/coef(mod)[2]); pow.set #-coef(mod)[1]
pow.set$date.time<-Sys.time()
print(pow.set[c(1,3)], digits = 2)

