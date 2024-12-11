#### id ####
library(wholebrain); library(data.table); library(ggplot2); library(gplots)

definitiveid<-read.table( text='  geno.y    animal ..1
                          1      WT R0060A150  82
                          2      WT R0060A161  46
                          3      WT R0060A167  88
                          4      WT R0060A180  74
                          5      WT R0060A710  74
                          6      WT R0060A715  72
                          7      WT R0060A918  82
                          8     HET R0060A146  84
                          9     HET R0060A152  70
                          10    HET R0060A185  82
                          11    HET R0060A913  72
                          12    HET R0060A915 112
                          13    HET R0060A919  82
                          14    HET R0060A920  74',header=TRUE, colClasses='character' )
definitiveid<-data.table(definitiveid)
definitiveid<-definitiveid[,c('t1','t2') := tstrsplit(animal, "A")]
definitiveid<-definitiveid$t2
