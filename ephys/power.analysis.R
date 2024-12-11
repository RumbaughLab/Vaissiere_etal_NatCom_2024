library(plyr)
library(ggplot2)
library(data.table)
dir<-'Y:/Vaissiere/00-ephys temp/';for(i in dir){
  setwd(i)
  setwd(i)
}

patchcableligth<-read.csv('patchcableligth.csv', header = TRUE)
str(patchcableligth)
current<-Sys.time()
ligthmeasure<-5.55
patchcableligthTEMP<-data.table(current,ligthmeasure)
patchcableligthTEMP<-rbind(patchcableligth, patchcableligthTEMP)
write.csv(patchcableligthTEMP, 'patchcableligth.csv', row.names = FALSE)


# since 2/5/2018 measured baseline 
# 3.638 from the probe
toadd1<-read.csv('power.set1.csv', colClasses = c(rep("numeric",4),"POSIXct"))
mat.lab.input<-c(0,0.1,1,2,3,4,5)
# old measured.value<-c(0,0.072,0.722,1.270,1.857,2.328,2.751) # the measured values are in mW
measured.value<-c(0,0.072,1.03,2.02,2.55,3.2,3.79) # the measured values are in mW


# power analysis extract coef
#toadd1<-read.csv('power.set1.csv', colClasses = c(rep("numeric",4),"POSIXct"))
#mat.lab.input<-c(0,0.1,1,2,3,4,5)
#measured.value<-c(0,0.098,0.395,0.737,1.035,1.300,1.537) # the measured values are in mW

#last meas for 10mm
#mat.lab.input<-c(0,0.1,1,2,3,4,5)
#measured.value<-c(0,0.046,0.464,0.861,1.204,1.510,1.786) # the measured values are in mW

val=0.35 
data.table(measured.value, measured.value-measured.value*val)

power.dat<-data.table(mat.lab.input,measured.value)

mod<-lm(measured.value ~ mat.lab.input, power.dat); summary(mod)

power.dat$intercept<-coef(mod)[1]
power.dat$slope<-coef(mod)[2]
power.dat$date.time<-Sys.time()


windows()
ggplot(power.dat, aes(x=mat.lab.input, y=measured.value, group = 1)) +
  geom_smooth(se = TRUE, method = "lm", color="black")+
  geom_point()+
  theme.tv

power.dat<-rbind(power.dat, toadd1)

write.csv(power.dat, "power.set1.csv", row.names = FALSE)

# series for desried power
toadd2<-read.csv('power.set2.csv', colClasses = c(rep("numeric",3),"POSIXct"))
#pow.set<-data.table(desired.mWmm2=c(0.5,1,5,10,25,50,75))
pow.set<-data.table(desired.mWmm2=c(0,1,5,10,15,20,25,30,35,40))
pow.set<-ddply(pow.set,
               .(desired.mWmm2),
               mutate,
               mW.equi = 0.1*0.1*pi*desired.mWmm2,
               toinput.inmat = mW.equi/coef(mod)[2]-coef(mod)[1])
pow.set$date.time<-Sys.time()
ab<-print(pow.set[c(1,3)], digits = 1)
pow.set<-rbind(pow.set, toadd2)
write.csv(pow.set, "power.set2.csv", row.names = FALSE)


# matlab equivalent for following values
mat.equi<-data.table(mat.val=c(1,2,3,4,5))
mat.equi<-ddply(mat.equi,
                .(mat.val),
                transform,
                mW.mm2=(mat.val*coef(mod)[2]+coef(mod)[1])/(0.1*0.1*pi))
mat.equi

# new power set ####
pow.set<-data.table(desired.mWmm2=c(0,1.4,2.8,4.2,5.6,7,8.4,10,20,40))
pow.set<-ddply(pow.set,
               .(desired.mWmm2),
               mutate,
               mW.equi = 0.1*0.1*pi*desired.mWmm2,
               toinput.inmat = mW.equi/coef(mod)[2]); pow.set #-coef(mod)[1]
pow.set$date.time<-Sys.time()
print(pow.set[c(1,3)], digits = 2)
pow.set<-rbind(pow.set, toadd2)
