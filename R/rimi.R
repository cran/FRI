#' Relative Importance of Main and Interaction Effects
#' @param data input data set
#' @description A new method to compute relative importance of main and interaction effects
#' of inputs in Artificial Neural Networks. The method was published in a paper on 20 June 2022
#' at <https://link.springer.com/article/10.1134/S1064229322080051> under the title of
#' "Modeling Main and Interactional Effects of Some Physiochemical Properties of Egyptian Soils
#' on Cation Exchange Capacity Via Artificial Neural Networks". The relative importance is computed
#' based on R square, and recomputed based on 100 percent for comparison. Also, sum of the modified
#' generalized weights is computed.
#' @usage rimi(data)
#' @return A table and figure with relative importance of inputs and their two way interaction
#' @references Ibrahim, O.M., El-Gamal, E.H., Darwish, K.M. Modeling Main and Interactional Effects
#' of Some Physiochemical Properties of Egyptian Soils on Cation Exchange Capacity Via Artificial
#' Neural Networks. Eurasian Soil Sc. (2022). https://doi.org/10.1134/S1064229322080051
#' @export
#' @import stats
#' @source <https://github.com/dromarnrc/Modified-Generalized-Weights/blob/main/MGW>
#' @examples x1<-rnorm(100,2,0.5)
#' x2<-rnorm(100,3,2)
#' y<-rnorm(100,6,3)
#' df<-data.frame(x1,x2,y)
#' rimi(df)
#' @details The data must be two or more numeric inputs and one output, the output must be in the last column,
#' columns must have headers or names. The used neural network is Multilayer perceptron with back propagation
#' algorithm. The number of neurons in hidden layer is 1.6 times the number of inputs. If you want to change
#' these setting, you can use the code on github.
rimi<-function(data){
Inp <- base::ncol(data) - 1
Nr_of_int <- Inp  * (Inp - 1)/2
AE <- Inp + Nr_of_int  # all effects (main + interaction)
NrHidden <- base::round(Inp * 1.6, 0)
partition <- 0.75  #** partitioning ratio of data into train and test
RI<-array(0,dim = c(AE,AE,3))
for (xx in 1:AE) {
  trainingIndex <- sample(1:base::nrow(data), partition * base::nrow(data))
  indx <- base::sample(1:base::nrow(data), 0.75 * base::nrow(data))
  trainingData <- data.frame(base::round(RSNNS::normalizeData(data[indx,],"0_1"),4))
  testingData <- data.frame(base::round(RSNNS::normalizeData(data[-indx,],"0_1"),4))
  names(trainingData)<-colnames(data)
  names(testingData)<- colnames(data)
  y <- colnames(trainingData[base::ncol(data)])
  x <- colnames(trainingData[-base::ncol(data)])
  formla <- stats::as.formula(paste(y, paste(x, collapse = " + "),sep = " ~ "))
  nn <- neuralnet::neuralnet(formla, data = trainingData, algorithm = "backprop", threshold = 0.01, learningrate = 0.01,
                             hidden = NrHidden, act.fct = "logistic", linear.output = F)
  IH_W <- nn$weights[[1]][[1]]
  HO_W <- nn$weights[[1]][[2]]
  # ***********************
  HU <- NrHidden
  # hold bias and input values (bias = 1)
  Inputs <- array(1, dim = c(length(trainingIndex), Inp + 1))
  for (k in 1:length(trainingIndex)) {
    for (l in 1:Inp) {
      Inputs[k, l + 1] <- trainingData[k, l]
    }
  }
  # calculate output of each hidden layer neuron
  HNO <- array(0, dim = c(length(trainingIndex), HU))
  Sigm <- array(0, dim = c(length(trainingIndex), HU))
  for (x in 1:HU) {
    for (y in 1:length(trainingIndex)) {
      for (z in 1:(Inp + 1)) {
        HNO[y, x] <- HNO[y, x] + (Inputs[y, z] * IH_W[z, x])
      }
    }
  }
  Sigm <- 1/(1 + exp(-(HNO)))
  # ************ Calculate generalized weights
  GWtable <- array(0, dim = c(length(trainingIndex),
                              base::ncol(data) - 1))
  colnames(GWtable) <- colnames(data[-(base::ncol(data))])
  for (a in 1:Inp) {
    for (b in 1:length(trainingIndex)) {
      for (c in 1:HU) {
        GWtable[b, a] <- GWtable[b, a] + (Sigm[b, c] * (1 - Sigm[b, c]) * IH_W[a + 1, c] * HO_W[c + 1])
      }
    }
  }
  # *************** Hidden outputs for Main effect
  MainE <- array(0, dim = c(length(trainingIndex), HU, Inp))
  for (q in 1:Inp) {
    for (r in 1:length(trainingIndex)) {
      for (h in 1:HU) {
        ssum = 0
        for (i in 2:(Inp + 1)) {
          ssum <- ssum + (Inputs[r, i] * IH_W[i, h])
          MainE[r, h, q] <- HNO[r, h] - ssum + (Inputs[r, q + 1] * IH_W[q + 1, h])
        }
      }
    }
  }
  SigMainE <- 1/(1 + exp(-(MainE)))
  # ************ generalized weights for main effect
  MainGW <- array(0, dim = c(length(trainingIndex), base::ncol(data) - 1))
  colnames(MainGW) <- colnames(data[-(base::ncol(data))])
  for (s in 1:Inp) {
    for (t in 1:length(trainingIndex)) {
      for (u in 1:HU) {
        MainGW[t, s] <- MainGW[t, s] + (SigMainE[t, u, s] * (1 - SigMainE[t, u, s]) * IH_W[s + 1, u] * HO_W[u + 1])
      }
    }
  }
  # ******** if inputs = 2 do the following
  if (Inp == 2) {
    pair<-list(c(1,2))
    FMGW <- as.data.frame(MainGW)
    MGW <- as.data.frame(GWtable)
    Nr_of_int <- (Inp - 0) * (Inp - 1)/2
    tableMEInt <- array(0, dim = c(3, (Inp + Nr_of_int)))
    intmgw <- array(0, dim = c(length(trainingIndex), 1))
    MESum <- c()
    for (p in 1:length(trainingIndex)) {
      MESum[p] <- FMGW[p, 1] + FMGW[p, 2]
      intmgw[p, 1] <- intmgw[p, 1] + ((MGW[p, 1]+ MGW[p, 2]) - MESum[p])/2
    }
    MGWMEInt <- cbind(FMGW, intmgw)
    totalvar = 0
    totalsum = 0
    for (n in 1:(Inp + Nr_of_int)) {
      totalvar <- totalvar + stats::var(MGWMEInt[, n])
      totalsum <- totalsum + abs(base::sum(MGWMEInt[, n]))
    }
    for (m in 1:3) {
      for (n in 1:(Inp + Nr_of_int)) {
        if (m == 1) {
          tableMEInt[m, n] <- base::round(abs(base::sum(MGWMEInt[, n]))/totalsum, 4)
        } else if (m == 2) {
          tableMEInt[m, n] <- base::round(stats::var(MGWMEInt[, n]), 4)
        } else {
          tableMEInt[m, n] <- round(base::sum(MGWMEInt[, n]), 4)
        }
      }
    }
    # ***** Print results for best runs
    mylist <- list()
    int<-paste(colnames(data[1]), colnames(data[2]), sep = "*")
    mylist <- c(colnames(data[1]),colnames(data[2]),int)
    rownames(tableMEInt) <- c("Rel.Imp. based on 100%", "Variance of MGW", "Sum of MGW")
    colnames(tableMEInt) <- mylist
    NN_Output <- stats::predict(nn,testingData)
    actual <- testingData[base::ncol(data)]
    df<-array(0,dim=c(1,AE))
    colnames(df) = mylist
    for (r in 1:AE) {
      df[1,r]<-round(tableMEInt[1,r]*(stats::cor(NN_Output,actual))^2,4)
    }
    RS<-round(cor(NN_Output,actual)^2,3)
    rownames(df)<-paste("Rel.Imp.based on R Sq.","(",RS,")", sep = "")
    df1<-rbind(df,tableMEInt)
    AllMainEffect<-rowSums(df1[,1:Inp])
    AllInteractionEffect<-df1[,3]
    df1<-cbind(df1,Total.M.E=AllMainEffect,Total.I.E=AllInteractionEffect)
    RI[xx,,1]<-df1[1,1:AE]
    RI[xx,,2]<-df1[2,1:AE]
    RI[xx,,3]<-df1[4,1:AE]
  } else {
    # ********* if inputs more than 2 do the following
    # ********* hidden output for two way interaction
    TwoWay <- array(0, dim = c(length(trainingIndex), HU, Inp * (Inp - 1)))
    serial<-c(2:(Inp+1))
    i=1
    s<-c()
    for (x in 2:(Inp+1)) {
      a<-subset(serial,serial!=x)
      for (y in a) {
        b <-subset(serial,serial!=x & serial!=y)
        for (z in rev(b)) {
          s[i]<-z
          i=i+1
        }
      }
    }
    for (q in 1:((Inp - 0) * (Inp - 1))) {
      for (r in 1:length(trainingIndex)) {
        ssum = 0
        for (h in 1:HU) {
          ssum <- ssum + (Inputs[r, s[q]] * IH_W[s[q], h])
          TwoWay[r, h, q] <- HNO[r, h] - ssum
        }
      }
    }
    SigTwoWay <- 1/(1 + exp(-(TwoWay)))
    # ********* GW for two way interaction
    TwoWayGW <- array(0, dim = c(length(trainingIndex), Inp * (Inp - 1)))
    i=1
    s<-c()
    for (x in 2:(Inp+1)) {
      for (y in 1:(Inp-1)) {
        s[i]<-x
        i=i+1
      }
    }
    for (q in 1:((Inp - 0) * (Inp - 1))) {
      for (t in 1:length(trainingIndex)) {
        for (u in 1:HU) {
          TwoWayGW[t, q] <- TwoWayGW[t, q] + (SigTwoWay[t, u, q] * (1 - SigTwoWay[t, u, q]) * IH_W[s[q], u] * HO_W[u + 1])
        }
      }
    }
    # ***** Main effect and Interaction based on sum
    tableMEInt <- array(0, dim = c(3, Inp + Nr_of_int))
    intmgw <- array(0, dim = c(length(trainingIndex), Nr_of_int))
    MESum <- array(0, dim = c(length(trainingIndex), Nr_of_int))
    i=1
    s<-c()
    pair<-array(0,dim=c((Inp*(Inp-1))/2,2))
    for (x in 1:(Inp-1)) {
      for (y in x:x) {
        for (z in (y+1):Inp) {
          pair[i,1]<-y
          pair[i,2]<-z
          i=i+1
        }
      }
    }
    for (x in 1:length(trainingIndex)) {
      for (y in 1:Nr_of_int) {
        MESum[x, y] <- MainGW[x, pair[y,1]] + MainGW[x, pair[y,2]]
      }
    }
    i=1
    s<-c(1:Inp)
    pairs<-as.data.frame(array(0,dim=c(Inp*(Inp-1),2)))
    for (x in 1:Inp) {
      for (y in x:x) {
        ss<-subset(s,s!=y)
        for (z in ss) {
          pairs[i,1]<-y
          pairs[i,2]<-z
          i=i+1
        }
      }
    }
    s<-as.data.frame(array(0,dim=c((Inp*(Inp-1)/2),2)))
    for (r in 1:(Inp*(Inp-1)/2)) {
      f<-which(pairs[,1] == pair[r,1] & pairs[,2] == pair[r,2] | pairs[,1] == pair[r,2] & pairs[,2] == pair[r,1], arr.ind=TRUE)
      s[r,1]<-f[1]
      s[r,2]<-f[2]
    }

    for (i in 1:Nr_of_int) {
      for (p in 1:length(trainingIndex)) {
        intmgw[p, i] <- intmgw[p, i] + ((TwoWayGW[p, s[i,1]] + TwoWayGW[p, s[i,2]]) - MESum[p, i])/2
      }
    }
    # 2Xf2E6aEU7n685eHEbXGYHrmWn2y7a62UWBrtZzodVdD
    MGWMEInt <- cbind(MainGW, intmgw)
    totalvar = 0
    totalsum = 0
    for (n in 1:(Inp + Nr_of_int)) {
      totalvar <- totalvar + stats::var(MGWMEInt[, n])
      totalsum <- totalsum + abs(base::sum(MGWMEInt[, n]))
    }
    for (m in 1:3) {
      for (n in 1:(Inp + Nr_of_int)) {
        if (m == 1) {
          tableMEInt[m, n] <- round(abs(base::sum(MGWMEInt[, n]))/totalsum, 4)
        } else if (m == 2) {
          tableMEInt[m, n] <- round(stats::var(MGWMEInt[, n]), 4)
        } else {
          tableMEInt[m, n] <- round(base::sum(MGWMEInt[, n]), 4)
        }
      }
    }
    # Print results for best runs
    mylist <- list()
    for (x in 1:Nr_of_int) {
      for (y in pair[x]) {
        mylist[x] <- paste(colnames(data[pair[x,1]]), colnames(data[pair[x,2]]), sep = "*")
      }
    }
    rownames(tableMEInt) <- c("Rel.Imp. based on 100%", "Variance of MGW", "Sum of MGW")
    colnames(tableMEInt) <- c(colnames(data[-(base::ncol(data))]), mylist)
    NN_Output <- stats::predict(nn,testingData)
    actual <- testingData[base::ncol(data)]
    df<-array(0,dim=c(1,AE))
    colnames(df)=c(colnames(data[-(base::ncol(data))]), mylist)
    for (r in 1:AE) {
      df[1,r]<-round(tableMEInt[1,r]*(stats::cor(NN_Output,actual))^2,4)
    }
    RS<-base::round(stats::cor(NN_Output,actual)^2,3)
    rownames(df)<-base::paste("Rel.Imp.based on R Sq.","(",RS,")", sep = "")
    df1<-base::rbind(df,tableMEInt)
    AllMainEffect<-base::rowSums(df1[,1:Inp])
    AllInteractionEffect<-base::rowSums(df1[,(Inp + 1):base::ncol(df1)])
    df1<-base::cbind(df1,Total.M.E=AllMainEffect,Total.I.E=AllInteractionEffect)
    RI[xx,,1]<-df1[1,1:AE]
    RI[xx,,2]<-df1[2,1:AE]
    RI[xx,,3]<-df1[4,1:AE]
  }
}
RIdf<-array(0,dim = c(5,AE))
RIdf[1,]<-apply(RI[,,1],2,base::mean)
RIdf[2,] <- apply(RI[,,1],2,stats::sd)/base::sqrt(AE)
RIdf[3,]<-apply(RI[,,2],2,base::mean)
RIdf[4,] <- apply(RI[,,2],2,stats::sd)/base::sqrt(AE)
RIdf[5,]<-apply(RI[,,3],2,base::mean)
rownames(RIdf) <- c(paste("RI sum to R.sq","(",round(base::sum(RIdf[1,]),3),")", sep = ""),
                    "Std Error","RI sum to 1","Std Error","Sum of MGW")
colnames(RIdf)<-colnames(df)
RIdf1<-t(RIdf)
print(RIdf1)
RIdf2<-as.data.frame(RIdf1)
colnames(RIdf2)<-c("RI.Rsq","SE.Rsq","RI.1","SE.RI1","MGW.Sum")

for (i in c(1,3)) {
p <- ggplot2::ggplot(RIdf2,ggplot2::aes(x = forcats::fct_inorder(rownames(RIdf2)),y = RIdf2[,i], fill = rownames(RIdf2))) +
  ggplot2::geom_bar(stat = "identity", position = "dodge") +
  ggplot2::geom_errorbar(ggplot2::aes(ymin =  RIdf2[,i]- RIdf2[,i+1], ymax = RIdf2[,i] + RIdf2[,i+1]), position = ggplot2::position_dodge(0.5), width = 0.2) +
  ggplot2::xlab(expression(Main~effects~and~interactions)) +
  ggplot2::ylab(expression(Relative~Importance)) +
  ggplot2::geom_text(ggplot2::aes(x = as.character(forcats::fct_inorder(rownames(RIdf2))),
                y = RIdf2[,i] + RIdf2[,i+1] + ((RIdf2[,i] + RIdf2[,i+1])*0.01), label = round(RIdf2[,5],2)),
            hjust = 0.5, vjust = -0.5, size = 4, colour = "red", fontface = "bold", angle = 360) +
  ggplot2::theme(axis.text.x = ggplot2::element_text(color = "blue", size = 10, angle = 30, vjust = 0.8, hjust = 0.8)) +
  ggplot2::theme(legend.position = "none")
    if (i == 1) {
  p = p + ggplot2::ggtitle(expression("Relative Importance sum to" ~ R^2))
  print(p)
    } else {
  p = p + ggplot2::ggtitle(expression("Relative Importance sum to 1"))
  print(p)
    }
}
}
