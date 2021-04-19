function R = Greedy_weighted_2(Requests, Edges, Edges_Models, Max_Delay, Max_Accuracy, Communication_Delays, AverageQueueDelay, w1, w2)
%Requests = zeros(n,5);    %time of request, the edge the request is coming from, requested accuracy,
                          %requested delay, preprocessing delay
n = length(Requests(:,1));
%User_Satisfaction = zeros(n,1);
iteration = 1;
counter = 0;
Results = zeros(n, 7); %,feasible, edge number, edge model, accuracy, delay
%Fitness_req = zeros(n,2); % req number, fitness_get
Fitness_edge = zeros(length(Edges_Models),6);               %edge number, fitness, acc, delay
while counter < iteration
    counter = counter + 1;
    %%choosing based on least Communication delay and accuracy
    for i = 1: length(Requests(:,1))
        source_req = Requests(i,2);
        counter_fitness = 0;
        for j = 1: length(Edges_Models(:,1))
            if Edges_Models(j,3) == Requests(i,1)... 
                && Edges_Models(j,5) >= Requests(i,3) && ((Edges_Models(j,6)...
                    + AverageQueueDelay(i) + Requests(i,5) ...
                    + Communication_Delays(source_req,Edges_Models(j,1))* 2) <= Requests(i,4))
                counter_fitness = counter_fitness + 1;
                Fitness_edge(counter_fitness,1) = Edges_Models(j,1);
                alfa = w1(i,1)*((Edges_Models(j,5)- Requests(i,3))/Max_Accuracy);
                beta = (w2(i,1)*(( Requests(i,4)- (Edges_Models(j,6)+ ...
                     (Communication_Delays(source_req,Edges_Models(j,1)) * 2) + AverageQueueDelay(i) + ...
                     Requests(i,5)))/Max_Delay));
                %Fitness_edge(counter_fitness,2) = 1/ (1 + exp(-(alfa + beta)));
%                 Fitness_edge(counter_fitness,2) = (w1(i,1)*((Edges_Models(j,5)- Requests(i,3))/Max_Accuracy )) +...
%                     (w2(i,1)*(( Requests(i,4)- (Edges_Models(j,6)+ ...
%                     (Communication_Delays(source_req,Edges_Models(j,1)) * 2) + AverageQueueDelay(i) + ...
%                 Requests(i,5)))/Max_Delay));
                if alfa >= 0 && beta >= 0
                    Fitness_edge(counter_fitness,2) = 1/ (1 + exp(-(alfa + beta)));
                else
                    Fitness_edge(counter_fitness,2) = -1/ (1 + exp(-(alfa + beta)));
                end
                Fitness_edge(counter_fitness,3) = Edges_Models(j,5);
                Fitness_edge(counter_fitness,4) = Edges_Models(j,6);
                Fitness_edge(counter_fitness,5) = w1(i,1)*((Edges_Models(j,5)- Requests(i,3))/Max_Accuracy );
                Fitness_edge(counter_fitness,6) = w2(i,1)*(( Requests(i,4)- (Edges_Models(j,6)+ ...
                    (Communication_Delays(source_req,Edges_Models(j,1)) * 2) + AverageQueueDelay(i) + ...
                Requests(i,5)))/Max_Delay);
            end
        end
        Fitness_edge = Fitness_edge(Fitness_edge(:,1)>0,:);
        Fitness_edge = sortrows(Fitness_edge,2,'descend');
        for j=1 : length(Fitness_edge(:,1))
            if Edges(Fitness_edge(j,1),3) >= 1 %...& Fitness_edge(j,3) >= Requests(i,3) ...
                    %...&& ((Fitness_edge(j,4) + AverageQueueDelay(i) + Requests(i,5) ...
                    %...+ Communication_Delays(source_req,Fitness_edge(j,1))* 2) <= Requests(i,4))
                if (Fitness_edge(j,1) ~= source_req) && Edges(source_req,2) >= 1
                   Results(i,2) = Fitness_edge(j,1);
                   Results(i,3) = Fitness_edge(j,2);
                   Results(i,4) = Fitness_edge(j,3);
                   Results(i,5) = Fitness_edge(j,4) ...
                        + AverageQueueDelay(i) + Requests(i, 5) ...
                        + Communication_Delays(source_req,Fitness_edge(j,1))* 2;
                   Results(i,6) = Fitness_edge(j,5);
                   Results(i,7) = Fitness_edge(j,6);
                   Edges(Fitness_edge(j,1),3) = Edges(Fitness_edge(j,1),3) - 1;
                   Edges(source_req,2) = Edges(source_req,2) - 1;
                   if Fitness_edge(j,1) == 10 %central cloud
                       Results(i,1) = 2;
                   else
                       Results(i,1) = 3;
                   end
                   break;
                elseif Fitness_edge(j,1) == source_req
                   Results(i,2) = Fitness_edge(j,1);
                   Results(i,1) = 1;
                   Results(i,3) = Fitness_edge(j,2);
                   Results(i,4) = Fitness_edge(j,3);
                   Results(i,5) = Fitness_edge(j,4) ...
                        + AverageQueueDelay(i) + Requests(i, 5);
                   Results(i,6) = Fitness_edge(j,5);
                   Results(i,7) = Fitness_edge(j,6);
                   Edges(source_req,3) = Edges(source_req,3) - 1;
                   break;
                end
               %break;
            end
        end
       
    end
%      for i = 1: length(Results(:,1))
%         if Results(i,1) > 0 & Results(i,3)~=0
%             User_Satisfaction(i) = (w1(i)*((Results(i,4) - Requests(i,3))/Max_Accuracy)) + (w2(i)*((Requests(i,4) - Results(i,5))/Max_Delay));
%         end
%     end
    %Satisfied = length(find(Results(:,1)>0));
    %Final_US = Satisfied/n;
    %R.Final_US = Final_US;
    Served = length(find(Results(:,1)>0));
    Final_Served = Served/n;
    R.Final_Served = Final_Served;
    Satisfied = ((length(find( Results(:,3)>= 0.5 ))) / n );
    R.Final_Satisfied = Satisfied;
    s = 0;
    for i = 1: n
        s = s + (Results(i,3) * (Results(i,1)>0));
    end
    s = s - ( n - Served);
    R.sum_US = (s/n);
    R.Loss = 1-Satisfied;
    R.Offloaded = (length(find(Results(:,1)==2)))/n;
    R.Offloaded_to_edges = (length(find(Results(:,1)==3)))/n;
    R.local = (length(find(Results(:,1)==1)))/n;
    R.drop = (length(find(Results(:,1)==0)))/n;
    R.Accuracy = mean(Results(:,6)); 
    R.Delay = mean(Results(:,7));
end
    
    %Results
end

