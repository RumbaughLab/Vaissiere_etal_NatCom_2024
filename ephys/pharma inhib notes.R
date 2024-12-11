# molarity concentration NaCl 1M ####

M<-1 # desired molarity in M
MW<- 58.44 # molecular weight
V<- 0.05 # desired volume in L

mass.g<-M*MW*V; mass.g


# molarity concentration Alexa Fluor 594 Cadaverine A30678 ####

m<-1e-3 # mass in g
MW<-806.94 #mol weight
n<-m/MW; n # mol

V<-n*1000; V # intended volumne for dilution in L
M<-n/V; M # molarity in M

# if dilute 1mg in 1.2mL will end up with 1mM solution
# to use 200 ul in loading pipette add 20 ul to have final at 100 uM (60 times)
# to use 200 ul in loading pipette add 2 ul to have final at 10 uM (600 times)

# injection parameters would be based on time of recordings if placed with patch pipette

# pharmacological inactivation ####
# in vivo 200 uM CNQX, 500 uM AP5 at 100nl 5 depth point reported dilution in ringers
# in vitro lab Emin 10 uM NBQX with 50 uM AP5 in slice
# NBQX Tocris 1044 1mM stock solution

# prepare AP5 NBQX cocktail
# for AP5
mass.g<-4.5e-3   # in g
MW<-197.13      # in g.mol-1
M<-10e-3         # in M
V<-mass.g/(M*MW); V # in L
V*1e+6 #in ul

# for NBQX
mass.g<-1.8e-3   # in g
MW<-434.29       # in g.mol-1
M<-10e-3         # in M
V<-mass.g/(M*MW); V # in L
V*1e+6 #in ul

# pharma used mix 500uM APX and 200uM NBQX