function R = MakeData_EdgeModels()
%Service_Models = [
Edges_Models = zeros(1000,6);
counter_array = 0;
type_of_services = 100;
type_of_models = 10;
flag_service = zeros(type_of_services,1);
%flag_models = zeros(type_of_models,1);

 for i = 1 : 9
    if i<=3
         number_of_services = randi([1,20]);      %%number of services for each edge
    elseif i>3 && i<=6
         number_of_services = randi([1,40]);      %%number of services for each edge
    else
         number_of_services = randi([1,60]);      %%number of services for each edge
    end
    %number_of_services = 100;
    %Produced_service = 1;
    Produced_service = zeros(number_of_services,1);
    j = 1;
    while j <= number_of_services
        %number_of_models = randi([1,type_of_models]);
        %number_of_models = randi([1,3]);
        number_of_models = 1;
        %Service_type = randi([1,number_of_services]);
        Service_type = j;
        %while ismember(Service_type,Produced_service)== 0
            %Produced_service(j) = Service_type;
             j = j + 1;
             k = 1;
             Produced_models = zeros(number_of_models,1);
            while k <= number_of_models 
                model_type = randi([1,type_of_models]);
                %while ismember(model_type,Produced_models)== 0
                    Produced_models(k) = model_type;
                    k = k + 1;
                    counter_array = counter_array + 1;
                    Edges_Models(counter_array,1) = i;
                    Edges_Models(counter_array,2) = 1;
                    Edges_Models(counter_array,3) = Service_type;
                    Edges_Models(counter_array,4) = model_type;
                %end
            end
        %end
    end
end
Edges_Models = Edges_Models(Edges_Models(:,1)>0,:);
for j=1:type_of_services
    for i=1:length(Edges_Models)
        if Edges_Models(i,3) == j
            flag_service(j)=1;
            break;
        end
    end
    if flag_service(j)==0
        size = length(Edges_Models(:,1)) + 1; 
        Edges_Models(size,1) = randi([1,9]);
        Edges_Models(size,2) = 1;
        Edges_Models(size,3) = j;
        model_type = randi([1,type_of_models]);
        Edges_Models(size,4) = model_type;
        flag_service(j) = 1;
    end
end
for i = 1: length(Edges_Models(:,1))
    if Edges_Models(i,4) <= 3
       Edges_Models(i,5) = randi([41,45]);
       Edges_Models(i,6) = randi([900,910]);
    elseif Edges_Models(i,4) > 3 & Edges_Models(i,4) <= 6
        Edges_Models(i,5) = randi([51,55]);
        Edges_Models(i,6) = randi([1000,1010]);
    elseif Edges_Models(i,4) > 6 & Edges_Models(i,4) <= 10
        Edges_Models(i,5) = randi([61,65]);
        Edges_Models(i,6) = randi([1100,1110]);
    end
end
s = 1;
for i= length(Edges_Models(:,1)) : length(Edges_Models(:,1)) + type_of_services
    Edges_Models(i,1) = 10;
    Edges_Models(i,2) = 0;
    Edges_Models(i,3) = s;
    Edges_Models(i,4) = 1;
    Edges_Models(i,5) = 85;
    Edges_Models(i,6) = 100;
    s = s + 1;
end
Edges_Models
%flag_service
end
            