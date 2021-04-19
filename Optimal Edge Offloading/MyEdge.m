%% Mobile Edge Computing Offloading 
%  author: Minoo Hosseinzadeh
clc, clear
opt = optimset('Display', 'none');

%% basic parameter settings (had better not change those paras)
%%Models Specifications
Models = [ 1 87; 2 78; 3 67];
%% Edge Specifications
Edges = [1 24 25; 2 24 25; 3 24 25; 4 26 28;5 26 28; 6 26 28;7 28 30;...
    8 28 30; 9 28 30; 10 1 1000];
Edges_Models = [
   1           1          13           6          56        1007
           1           1          13           7          65        1103
           1           1          10           4          55        1008
           1           1           5           6          56        1000
           1           1           5           4          55        1010
           1           1           5           8          65        1106
           1           1          11           7          66        1101
           1           1           3           8          66        1108
           1           1           6           6          55        1000
           1           1           2           5          56        1002
           1           1           2           3          46         910
           1           1           4           3          45         900
           1           1           7           2          45         905
           1           1           1           1          46         909
           1           1           9           3          46         908
           1           1           9           8          66        1109
           1           1           9           1          45         901
           1           1           8           9          66        1107
           1           1           8           6          55        1005
           1           1           8          10          66        1109
           2           1           5           1          45         908
           2           1           5           8          66        1100
           2           1           7           8          66        1107
           2           1           1           2          45         908
           2           1           1           4          56        1000
           2           1           1          10          66        1108
           2           1           6           3          45         908
           2           1           6           2          46         900
           2           1           6           6          55        1007
           2           1           4           2          45         906
           2           1           3           2          46         906
           2           1           3           1          46         910
           2           1           3           8          65        1108
           2           1           2           2          45         903
           3           1           1           4          55        1006
           3           1           1           1          45         903
           3           1           2          10          65        1103
           3           1           2           9          66        1106
           4           1           1           1          46         906
           5           1          14           8          65        1105
           5           1          14           5          55        1003
           5           1          14           4          55        1003
           5           1          10           1          45         908
           5           1          13           3          46         903
           5           1           5           3          46         910
           5           1           5           1          46         905
           5           1           5           7          66        1103
           5           1          11           7          65        1100
           5           1          11          10          65        1110
           5           1          11           6          56        1006
           5           1          15           8          65        1105
           5           1          16           6          55        1008
           5           1          16           9          65        1110
           5           1          16           2          45         910
           5           1          18           4          56        1008
           5           1           6           2          45         906
           5           1           3          10          65        1105
           5           1           3           1          46         908
           5           1           4          10          65        1109
           5           1           4           3          45         906
           5           1           4           5          55        1003
           5           1           8           7          66        1102
           5           1           8           1          45         904
           5           1           8           6          55        1001
           5           1          17           8          65        1100
           5           1          17           2          46         909
           5           1          17           3          45         900
           5           1           2           6          56        1003
           5           1           2           7          66        1106
           5           1           9           4          55        1005
           5           1           9           9          65        1109
           5           1           9           1          45         907
           5           1           7           8          66        1106
           5           1           7           3          46         905
           5           1           7           1          46         900
           5           1          12           5          56        1007
           5           1          12          10          65        1101
           5           1           1           4          56        1002
           6           1           5           8          66        1107
           6           1           5           2          46         906
           6           1           8          10          65        1108
           6           1           8           1          45         909
           6           1           1           7          65        1104
           6           1           1           4          55        1010
           6           1           1           1          45         901
           6           1           4           6          56        1006
           6           1           9           7          65        1105
           6           1           9           4          55        1000
           6           1           3           1          46         904
           6           1           3           6          55        1001
           6           1           3           4          55        1008
           6           1           6           8          65        1105
           6           1           6           1          46         902
           6           1           6           3          45         902
           6           1           7           1          46         904
           6           1           2           5          55        1006
           6           1           2           3          45         905
           6           1           2           9          65        1107
           7           1          18           8          65        1101
           7           1          18          10          65        1104
           7           1          18           4          56        1006
           7           1           9           5          55        1009
           7           1           7           9          65        1100
           7           1          15           4          56        1006
           7           1          12          10          65        1106
           7           1          12           7          66        1107
           7           1           5           4          55        1009
           7           1           5          10          65        1103
           7           1          10           2          46         900
           7           1          13           2          45         910
           7           1          13           1          45         907
           7           1          16          10          66        1104
           7           1           3           3          45         910
           7           1          19           6          56        1004
           7           1          19           3          45         901
           7           1          19           8          65        1105
           7           1          11           5          55        1007
           7           1           6           5          56        1009
           7           1           6           4          55        1007
           7           1           1           6          56        1000
           7           1           1           8          65        1109
           7           1           2          10          65        1106
           7           1           2           7          65        1102
           7           1           8           2          46         910
           7           1          17          10          65        1110
           7           1          17           6          55        1008
           7           1          17           1          46         907
           7           1          14           4          56        1008
           7           1          14          10          65        1102
           7           1           4           7          66        1106
           7           1           4           6          55        1005
           8           1          14           7          65        1101
           8           1          14           3          45         908
           8           1          14           6          55        1000
           8           1          30           5          56        1000
           8           1          30           1          45         909
           8           1           6           7          66        1105
           8           1           6           3          46         902
           8           1           6           9          65        1105
           8           1          41           6          55        1003
           8           1           2           4          56        1002
           8           1           2           1          45         907
           8           1           3           5          55        1001
           8           1          40           4          56        1008
           8           1          40           9          65        1106
           8           1          40          10          66        1104
           8           1          18           5          56        1004
           8           1          36           9          65        1100
           8           1          36           1          46         902
           8           1          36           7          66        1102
           8           1           5           5          56        1009
           8           1           5          10          66        1102
           8           1           5           7          65        1107
           8           1          23           7          66        1107
           8           1          32           2          45         906
           8           1          32           5          55        1003
           8           1          50           7          65        1103
           8           1          50           2          45         903
           8           1          50           5          56        1009
           8           1          11           8          66        1100
           8           1          29           1          45         908
           8           1          17           9          66        1110
           8           1          46           1          46         908
           8           1          22           7          65        1105
           8           1          22           6          56        1010
           8           1          13           9          65        1102
           8           1          13           1          45         903
           8           1          13           3          45         909
           8           1          45           7          65        1100
           8           1          45           4          56        1001
           8           1          45           2          45         910
           8           1          28           2          45         908
           8           1           1           6          56        1003
           8           1          51           9          65        1107
           8           1          51           6          55        1005
           8           1          51           1          45         902
           8           1          39           5          56        1000
           8           1          39           3          46         905
           8           1          10           2          45         900
           8           1          37          10          66        1100
           8           1          37           1          45         901
           8           1          24           6          55        1005
           8           1          24           9          66        1109
           8           1          24           4          56        1005
           8           1          33           9          65        1103
           8           1          33           6          55        1007
           8           1          25           8          65        1103
           8           1          25          10          65        1101
           8           1          48           3          46         905
           8           1          48           7          65        1109
           8           1          52           2          45         909
           8           1          52           4          55        1002
           8           1          27           8          66        1110
           8           1          27          10          66        1105
           8           1          27           2          45         904
           8           1          12          10          65        1108
           8           1          12           7          66        1101
           8           1          35           1          46         900
           8           1          35           7          65        1104
           8           1          35           8          65        1104
           8           1          34           3          45         905
           8           1          31          10          65        1101
           8           1          47           8          65        1107
           8           1          47           7          65        1109
           8           1          47           2          46         902
           8           1          21           8          65        1110
           8           1          43           8          66        1103
           8           1           8           7          66        1102
           8           1          20           2          45         902
           8           1          26           9          66        1110
           8           1          26           3          46         904
           8           1          26          10          66        1109
           8           1          44           4          55        1000
           8           1          44           8          66        1108
           8           1           9          10          66        1100
           8           1           9           9          65        1102
           8           1          42           1          46         910
           8           1          42           7          65        1103
           8           1          42           6          55        1007
           8           1           4          10          65        1101
           8           1           7           7          65        1106
           8           1          15           9          65        1102
           8           1          15           1          45         909
           8           1          15           5          55        1009
           8           1          38           7          66        1104
           8           1          19           9          66        1106
           8           1          16           9          65        1100
           8           1          49           4          55        1007
           8           1          49           6          55        1008
           8           1          49           7          66        1104
           9           1          16           8          66        1108
           9           1          26           5          55        1005
           9           1          19           5          55        1008
           9           1          19          10          66        1108
           9           1          19           6          55        1000
           9           1           4          10          66        1109
           9           1           4           1          46         904
           9           1           2           5          56        1006
           9           1           2           1          46         903
           9           1           2          10          65        1103
           9           1           5          10          65        1101
           9           1           5           2          46         904
           9           1          28           4          55        1002
           9           1          28           6          55        1004
           9           1          15           3          45         910
           9           1          15           6          56        1000
           9           1          30           1          45         906
           9           1          30           9          66        1104
           9           1          30           3          46         905
           9           1          21           6          56        1003
           9           1          10           1          46         906
           9           1          29           3          46         904
           9           1          13           3          45         905
           9           1          13           6          56        1008
           9           1          13           5          56        1009
           9           1          25           6          55        1010
           9           1          25           5          56        1010
           9           1          25           1          46         900
           9           1          17           3          45         906
           9           1          17          10          65        1104
           9           1           6           8          65        1101
           9           1           6           1          45         903
           9           1          24           6          56        1006
           9           1          22           9          65        1102
           9           1          22           2          46         904
           9           1          11           4          55        1006
           9           1           1           8          65        1105
           9           1          14           3          46         906
           9           1          14           8          65        1105
           9           1           9           7          66        1104
           9           1           9           3          45         905
           9           1           9           2          45         906
           9           1          23           8          65        1109
           9           1          23           5          56        1006
           9           1          23           1          45         910
           9           1           3           6          55        1003
           9           1           3          10          66        1107
           9           1           3           1          46         903
           9           1          18          10          66        1104
           9           1          12           2          45         909
           9           1          12           1          45         907
           9           1          27           6          56        1001
           9           1          27           5          55        1005
           9           1          27           2          45         908
           9           1           7           3          46         908
           9           1           7          10          66        1102
           9           1          20           2          46         901
           9           1          20           7          66        1107
           9           1           8           9          66        1102
           9           1           8           1          46         901
           1           1          53           7          66        1105
           2           1          54           9          66        1101
           4           1          55           3          45         904
           3           1          56           7          65        1104
           1           1          57           5          56        1004
           1           1          58           2          45         905
           9           1          59           5          55        1006
           5           1          60           7          65        1110
           1           1          61           9          66        1104
           9           1          62           9          65        1106
           8           1          63           4          56        1008
           4           1          64           8          65        1110
           2           1          65           3          45         905
           2           1          66           7          65        1103
           7           1          67           7          65        1101
           7           1          68           3          46         907
           2           1          69           7          66        1102
           8           1          70           2          45         904
           5           1          71           9          66        1106
           7           1          72          10          65        1105
           8           1          73           5          55        1006
           9           1          74           8          65        1100
           5           1          75           6          55        1009
           5           1          76           3          46         910
           7           1          77           6          56        1008
           5           1          78          10          65        1110
           7           1          79           1          45         904
           8           1          80           6          55        1004
           8           1          81           5          56        1003
           3           1          82           8          65        1102
           3           1          83           2          46         907
           4           1          84           8          65        1100
           1           1          85           3          45         902
           1           1          86           5          55        1006
           8           1          87           5          56        1006
           8           1          88           3          46         906
           4           1          89           6          55        1010
           2           1          90           6          56        1002
           6           1          91           7          66        1107
           2           1          92           7          66        1104
           2           1          93           6          56        1008
           6           1          94           1          45         900
           3           1          95           2          45         900
           2           1          96           3          45         900
           3           1          97           8          66        1102
           9           1          98           9          66        1102
           1           1          99           3          45         908
          10           0           1           1          85         100
          10           0           2           1          85         100
          10           0           3           1          85         100
          10           0           4           1          85         100
          10           0           5           1          85         100
          10           0           6           1          85         100
          10           0           7           1          85         100
          10           0           8           1          85         100
          10           0           9           1          85         100
          10           0          10           1          85         100
          10           0          11           1          85         100
          10           0          12           1          85         100
          10           0          13           1          85         100
          10           0          14           1          85         100
          10           0          15           1          85         100
          10           0          16           1          85         100
          10           0          17           1          85         100
          10           0          18           1          85         100
          10           0          19           1          85         100
          10           0          20           1          85         100
          10           0          21           1          85         100
          10           0          22           1          85         100
          10           0          23           1          85         100
          10           0          24           1          85         100
          10           0          25           1          85         100
          10           0          26           1          85         100
          10           0          27           1          85         100
          10           0          28           1          85         100
          10           0          29           1          85         100
          10           0          30           1          85         100
          10           0          31           1          85         100
          10           0          32           1          85         100
          10           0          33           1          85         100
          10           0          34           1          85         100
          10           0          35           1          85         100
          10           0          36           1          85         100
          10           0          37           1          85         100
          10           0          38           1          85         100
          10           0          39           1          85         100
          10           0          40           1          85         100
          10           0          41           1          85         100
          10           0          42           1          85         100
          10           0          43           1          85         100
          10           0          44           1          85         100
          10           0          45           1          85         100
          10           0          46           1          85         100
          10           0          47           1          85         100
          10           0          48           1          85         100
          10           0          49           1          85         100
          10           0          50           1          85         100
          10           0          51           1          85         100
          10           0          52           1          85         100
          10           0          53           1          85         100
          10           0          54           1          85         100
          10           0          55           1          85         100
          10           0          56           1          85         100
          10           0          57           1          85         100
          10           0          58           1          85         100
          10           0          59           1          85         100
          10           0          60           1          85         100
          10           0          61           1          85         100
          10           0          62           1          85         100
          10           0          63           1          85         100
          10           0          64           1          85         100
          10           0          65           1          85         100
          10           0          66           1          85         100
          10           0          67           1          85         100
          10           0          68           1          85         100
          10           0          69           1          85         100
          10           0          70           1          85         100
          10           0          71           1          85         100
          10           0          72           1          85         100
          10           0          73           1          85         100
          10           0          74           1          85         100
          10           0          75           1          85         100
          10           0          76           1          85         100
          10           0          77           1          85         100
          10           0          78           1          85         100
          10           0          79           1          85         100
          10           0          80           1          85         100
          10           0          81           1          85         100
          10           0          82           1          85         100
          10           0          83           1          85         100
          10           0          84           1          85         100
          10           0          85           1          85         100
          10           0          86           1          85         100
          10           0          87           1          85         100
          10           0          88           1          85         100
          10           0          89           1          85         100
          10           0          90           1          85         100
          10           0          91           1          85         100
          10           0          92           1          85         100
          10           0          93           1          85         100
          10           0          94           1          85         100
          10           0          95           1          85         100
          10           0          96           1          85         100
          10           0          97           1          85         100
          10           0          98           1          85         100
          10           0          99           1          85         100
          10           0         100           1          85         100         
];

Communication_Delays = [0 1426 1427 1434 1432 1432 1436 1451 1438 6006;...
                        1426 0 1427 1433 1426 1428 1434 1435 1435 6007;...
                        1427 1427 0 1426 1429 1425 1427 1434 1429 6005;...
                        1434 1423 1426 0 1435 1427 1425 1437 1431 6007;...
                        1432 1426 1429 1435 0 1427 1433 1429 1429 6200;...
                        1432 1428 1425 1427 1427 0 1426 1429 1424 6200;...
                        1436 1434 1427 1425 1433 1426 0 1436 1428 6200;...
                        1451 1445 1434 1437 1429 1429 1436 0 1428 6400;...
                        1438 1435 1429 1431 1429 1424 1428 1428 0 6400;...
                        6006 6007 6005 6007 6200 6200 6200 6400 6400 0];
%% simulation begin
Max_Accuracy = 100;
%Max_Accuracy = 85;
Max_Delay = 12000;
total_run = 1000;
%n = 10;
counter_col = 34;
Total_Result_satisfied_p_Greedy = zeros(total_run,counter_col);
Total_Result_satisfied_f_Greedy = zeros(total_run,counter_col);
Total_Result_loss_Greedy = zeros(total_run,counter_col);
Total_Result_offloaded_Greedy = zeros(total_run,counter_col);
Total_Result_offloaded_toEdge_Greedy = zeros(total_run,counter_col);
Total_Result_local_Greedy = zeros(total_run,counter_col);
Total_Result_dropped_Greedy = zeros(total_run,counter_col);
Total_Result_Accuracy_Greedy = zeros(total_run,counter_col);
Total_Result_Delay_Greedy = zeros(total_run,counter_col);
Total_Result_Served_Greedy = zeros(total_run,counter_col);

Total_Result_satisfied_p_Random = zeros(total_run,counter_col);
Total_Result_satisfied_f_Random = zeros(total_run,counter_col);
Total_Result_loss_Random = zeros(total_run,counter_col);
Total_Result_offloaded_Random = zeros(total_run,counter_col);
Total_Result_offloaded_toEdge_Random = zeros(total_run,counter_col);
Total_Result_local_Random = zeros(total_run,counter_col);
Total_Result_dropped_Random = zeros(total_run,counter_col);
Total_Result_Served_Random = zeros(total_run,counter_col);

Total_Result_satisfied_p_OffloadAll = zeros(total_run,counter_col);
Total_Result_satisfied_f_OffloadAll = zeros(total_run,counter_col);
Total_Result_loss_OffloadAll = zeros(total_run,counter_col);
Total_Result_offloaded_OffloadAll = zeros(total_run,counter_col);
Total_Result_offloaded_toEdge_OffloadAll = zeros(total_run,counter_col);
Total_Result_local_OffloadAll = zeros(total_run,counter_col);
Total_Result_dropped_OffloadAll = zeros(total_run,counter_col);
Total_Result_Served_OffloadAll = zeros(total_run,counter_col);

Total_Result_satisfied_p_Local = zeros(total_run,counter_col);
Total_Result_satisfied_f_Local = zeros(total_run,counter_col);
Total_Result_loss_Local = zeros(total_run,counter_col);
Total_Result_offloaded_Local = zeros(total_run,counter_col);
Total_Result_offloaded_toEdge_Local = zeros(total_run,counter_col);
Total_Result_local_Local = zeros(total_run,counter_col);
Total_Result_dropped_Local = zeros(total_run,counter_col);
Total_Result_Served_Local = zeros(total_run,counter_col);

Total_Result_satisfied_p_Happy = zeros(total_run,counter_col);
Total_Result_satisfied_f_Happy = zeros(total_run,counter_col);
Total_Result_loss_Happy = zeros(total_run,counter_col);
Total_Result_offloaded_Happy = zeros(total_run,counter_col);
Total_Result_offloaded_toEdge_Happy = zeros(total_run,counter_col);
Total_Result_local_Happy = zeros(total_run,counter_col);
Total_Result_dropped_Happy = zeros(total_run,counter_col);
Total_Result_Served_Happy = zeros(total_run,counter_col);

Total_Result_satisfied_p_Happy_Comp = zeros(total_run,counter_col);
Total_Result_satisfied_f_Happy_Comp = zeros(total_run,counter_col);
Total_Result_loss_Happy_Comp = zeros(total_run,counter_col);
Total_Result_offloaded_Happy_Comp = zeros(total_run,counter_col);
Total_Result_offloaded_toEdge_Happy_Comp = zeros(total_run,counter_col);
Total_Result_local_Happy_Comp = zeros(total_run,counter_col);
Total_Result_dropped_Happy_Comp = zeros(total_run,counter_col);
Total_Result_Served_Happy_Comp = zeros(total_run,counter_col);


Total_Result_satisfied_p_Random2 = zeros(total_run,counter_col);
Total_Result_satisfied_f_Random2 = zeros(total_run,counter_col);
Total_Result_loss_Random2 = zeros(total_run,counter_col);
Total_Result_offloaded_Random2 = zeros(total_run,counter_col);
Total_Result_local_Random2 = zeros(total_run,counter_col);
Total_Result_dropped_Random2 = zeros(total_run,counter_col);
Total_Result_Served_Random2 = zeros(total_run,counter_col);
%t = 1;
inner_run_counter = 20000;
start_std = 10;
std = 10;
std_max = 9000;
counter_std = 1;
%n = 200;
n = 1000;
%n_prime = 50;
counter_n = 1;
std_step = 300;
X_label = 'Accuracy Mean';
number_of_services = 99;
m = 10;                    %number of edge servers
while n <= total_run & std <= std_max
    %n = std;
    Result_of_30_Greedy = zeros(inner_run_counter,10);
    Result_of_30_Random = zeros(inner_run_counter,8);
    Result_of_30_OffloadAll = zeros(inner_run_counter,8);
    Result_of_30_Local = zeros(inner_run_counter,8);
    Result_of_30_Happy = zeros(inner_run_counter,8);
    Result_of_30_Happy_Comp = zeros(inner_run_counter,8);
    Result_of_30_Random2 = zeros(inner_run_counter,8);
    
    counter = 1;
    while counter <= inner_run_counter
    %disp(['===> Time slot #', num2str(t), ' <==='])
        Requests = zeros(n,5);
        AverageQueueDelay = zeros(n,1);
        w1 = zeros(n,1);
        w2 = zeros(n,2);
        %% initialization
        % generate the task request
        for i = 1 : n
            Requests(i,1) = randi([1,number_of_services]);
            Requests(i,2) = randi([1,m-1]);
            
            requested_accuracy = normrnd(45,10);
            while requested_accuracy <= 0 || requested_accuracy > 100
                requested_accuracy = normrnd(45,10);
            end
            Requests(i,3) = requested_accuracy;
            
            requested_delay = normrnd(1000,4000); %it was the best between
            
            while requested_delay <= 0 
                requested_delay = normrnd(1000,4000);
            end
            Requests(i,4) = requested_delay;
            Requests(i,5) = randi(5);
            
            AverageQueueDelay(i,1) = randi([std, std + std_step]);
            
            w2(i,1) = 1;
            w1(i,1) = 1;
        end  
        
        
        Result_Happy = Happy(Requests, Edges, Edges_Models, Max_Delay, Max_Accuracy, Communication_Delays, AverageQueueDelay, w1, w2);
        Result_of_30_Happy(counter,1) = Result_Happy.Final_Satisfied;
        Result_of_30_Happy(counter,2) = Result_Happy.sum_US;
        Result_of_30_Happy(counter,3) = Result_Happy.Loss;
        Result_of_30_Happy(counter,4) = Result_Happy.Offloaded;
        Result_of_30_Happy(counter,5) = Result_Happy.local;
        Result_of_30_Happy(counter,6) = Result_Happy.drop;
        Result_of_30_Happy(counter,7) = Result_Happy.Offloaded_to_edges;
        Result_of_30_Happy(counter,8) = Result_Happy.Final_Served;
        
        Result_Happy_Comp = Happy_Comp(Requests, Edges, Edges_Models, Max_Delay, Max_Accuracy, Communication_Delays, AverageQueueDelay, w1, w2);
        Result_of_30_Happy_Comp(counter,1) = Result_Happy_Comp.Final_Satisfied;
        Result_of_30_Happy_Comp(counter,2) = Result_Happy_Comp.sum_US;
        Result_of_30_Happy_Comp(counter,3) = Result_Happy_Comp.Loss;
        Result_of_30_Happy_Comp(counter,4) = Result_Happy_Comp.Offloaded;
        Result_of_30_Happy_Comp(counter,5) = Result_Happy_Comp.local;
        Result_of_30_Happy_Comp(counter,6) = Result_Happy_Comp.drop;
        Result_of_30_Happy_Comp(counter,7) = Result_Happy_Comp.Offloaded_to_edges;
        Result_of_30_Happy_Comp(counter,8) = Result_Happy_Comp.Final_Served;


        
        Result_Local = Local(Requests, Edges, Edges_Models, Max_Delay, Max_Accuracy, Communication_Delays, AverageQueueDelay, w1, w2);
        Result_of_30_Local(counter,1) = Result_Local.Final_Satisfied;
        Result_of_30_Local(counter,2) = Result_Local.sum_US;
        Result_of_30_Local(counter,3) = Result_Local.Loss;
        Result_of_30_Local(counter,4) = Result_Local.Offloaded;
        Result_of_30_Local(counter,5) = Result_Local.local;
        Result_of_30_Local(counter,6) = Result_Local.drop;
        Result_of_30_Local(counter,7) = Result_Local.Offloaded_to_edges;
        Result_of_30_Local(counter,8) = Result_Local.Final_Served;

        Result_Offload = Offload_All(Requests, Edges, Edges_Models, Max_Delay, Max_Accuracy, Communication_Delays, AverageQueueDelay, w1, w2);
        Result_of_30_OffloadAll(counter,1) = Result_Offload.Final_Satisfied;
        Result_of_30_OffloadAll(counter,2) = Result_Offload.sum_US;
        Result_of_30_OffloadAll(counter,3) = Result_Offload.Loss;
        Result_of_30_OffloadAll(counter,4) = Result_Offload.Offloaded;
        Result_of_30_OffloadAll(counter,5) = Result_Offload.local;
        Result_of_30_OffloadAll(counter,6) = Result_Offload.drop;
        Result_of_30_OffloadAll(counter,7) = Result_Offload.Offloaded_to_edges;
        Result_of_30_OffloadAll(counter,8) = Result_Offload.Final_Served;
        
        Result_Greedy = Greedy_weighted_2(Requests, Edges, Edges_Models, Max_Delay, Max_Accuracy, Communication_Delays, AverageQueueDelay, w1, w2);
        Result_of_30_Greedy(counter,1) = Result_Greedy.Final_Satisfied;
        Result_of_30_Greedy(counter,2) = Result_Greedy.sum_US;
        Result_of_30_Greedy(counter,3) = Result_Greedy.Loss;
        Result_of_30_Greedy(counter,4) = Result_Greedy.Offloaded;
        Result_of_30_Greedy(counter,5) = Result_Greedy.local;
        Result_of_30_Greedy(counter,6) = Result_Greedy.drop;
        Result_of_30_Greedy(counter,7) = Result_Greedy.Offloaded_to_edges;
        Result_of_30_Greedy(counter,8) = Result_Greedy.Accuracy;
        Result_of_30_Greedy(counter,9) = Result_Greedy.Delay;
        Result_of_30_Greedy(counter,10) = Result_Greedy.Final_Served;
        
        Result_Random = Random(Requests, Edges, Edges_Models, Max_Delay, Max_Accuracy, Communication_Delays, AverageQueueDelay, w1, w2);
        Result_of_30_Random(counter,1) = Result_Random.Final_Satisfied;
        Result_of_30_Random(counter,2) = Result_Random.sum_US;
        Result_of_30_Random(counter,3) = Result_Random.Loss;
        Result_of_30_Random(counter,4) = Result_Random.Offloaded;
        Result_of_30_Random(counter,5) = Result_Random.local;
        Result_of_30_Random(counter,6) = Result_Random.drop;
        Result_of_30_Random(counter,7) = Result_Random.Offloaded_to_edges;
        Result_of_30_Random(counter,8) = Result_Random.Final_Served;
        
        counter = counter + 1;
    end
    %n = n + 1;
%     temp_r2 =  mean(Result_of_30_Random2,1);
%     Total_Result_satisfied_p_Random2(n,counter_std) = temp_r2(1);
%     Total_Result_satisfied_f_Random2(n,counter_std) = temp_r2(2);
%     Total_Result_loss_Random2(n,counter_std) = temp_r2(3);
%     Total_Result_offloaded_Random2(n,counter_std) = temp_r2(4);
%     Total_Result_local_Random2(n,counter_std) = temp_r2(5);
%     Total_Result_dropped_Random2(n,counter_std) = temp_r2(6);
    %Total_Result_offloaded_toEdge_Random2(n,counter_std) = temp_r2(7);
    
    temp =  mean(Result_of_30_Happy,1);
    Total_Result_satisfied_p_Happy(1,counter_std) = temp(1);
    Total_Result_satisfied_f_Happy(1,counter_std) = temp(2);
    Total_Result_loss_Happy(1,counter_std) = temp(3);
    Total_Result_offloaded_Happy(1,counter_std) = temp(4);
    Total_Result_local_Happy(1,counter_std) = temp(5);
    Total_Result_dropped_Happy(1,counter_std) = temp(6);
    Total_Result_offloaded_toEdge_Happy(1,counter_std) = temp(7);
    Total_Result_Served_Happy(1,counter_std) = temp(8);
    
    temp =  mean(Result_of_30_Happy_Comp,1);
    Total_Result_satisfied_p_Happy_Comp(1,counter_std) = temp(1);
    Total_Result_satisfied_f_Happy_Comp(1,counter_std) = temp(2);
    Total_Result_loss_Happy_Comp(1,counter_std) = temp(3);
    Total_Result_offloaded_Happy_Comp(1,counter_std) = temp(4);
    Total_Result_local_Happy_Comp(1,counter_std) = temp(5);
    Total_Result_dropped_Happy_Comp(1,counter_std) = temp(6);
    Total_Result_offloaded_toEdge_Happy_Comp(1,counter_std) = temp(7);
    Total_Result_Served_Happy_Comp(1,counter_std) = temp(8);
    
    temp =  mean(Result_of_30_Greedy,1);
    Total_Result_satisfied_p_Greedy(1,counter_std) = temp(1);
    Total_Result_satisfied_f_Greedy(1,counter_std) = temp(2);
    Total_Result_loss_Greedy(1,counter_std) = temp(3);
    Total_Result_offloaded_Greedy(1,counter_std) = temp(4);
    Total_Result_local_Greedy(1,counter_std) = temp(5);
    Total_Result_dropped_Greedy(1,counter_std) = temp(6);
    Total_Result_offloaded_toEdge_Greedy(1,counter_std) = temp(7);
    Total_Result_Accuracy_Greedy(1,counter_std) = temp(8);
    Total_Result_Delay_Greedy(1,counter_std) = temp(9);
    Total_Result_Served_Greedy(1,counter_std) = temp(10);
    
    temp_r =  mean(Result_of_30_Random,1);
    Total_Result_satisfied_p_Random(1,counter_std) = temp_r(1);
    Total_Result_satisfied_f_Random(1,counter_std) = temp_r(2);
    Total_Result_loss_Random(1,counter_std) = temp_r(3);
    Total_Result_offloaded_Random(1,counter_std) = temp_r(4);
    Total_Result_local_Random(1,counter_std) = temp_r(5);
    Total_Result_dropped_Random(1,counter_std) = temp_r(6);
    Total_Result_offloaded_toEdge_Random(1,counter_std) = temp_r(7);
    Total_Result_Served_Random(1,counter_std) = temp_r(8);
    
    temp_o =  mean(Result_of_30_OffloadAll,1);
    Total_Result_satisfied_p_OffloadAll(1,counter_std) = temp_o(1);
    Total_Result_satisfied_f_OffloadAll(1,counter_std) = temp_o(2);
    Total_Result_loss_OffloadAll(1,counter_std) = temp_o(3);
    Total_Result_offloaded_OffloadAll(1,counter_std) = temp_o(4);
    Total_Result_local_OffloadAll(1,counter_std) = temp_o(5);
    Total_Result_dropped_OffloadAll(1,counter_std) = temp_o(6);
    Total_Result_offloaded_toEdge_OffloadAll(1,counter_std) = temp_o(7);
    Total_Result_Served_OffloadAll(1,counter_std) = temp_o(8);
    
    
    temp_l =  mean(Result_of_30_Local,1);
    Total_Result_satisfied_p_Local(1,counter_std) = temp_l(1);
    Total_Result_satisfied_f_Local(1,counter_std) = temp_l(2);
    Total_Result_loss_Local(1,counter_std) = temp_l(3);
    Total_Result_offloaded_Local(1,counter_std) = temp_l(4);
    Total_Result_local_Local(1,counter_std) = temp_l(5);
    Total_Result_dropped_Local(1,counter_std) = temp_l(6);
    Total_Result_offloaded_toEdge_Local(1,counter_std) = temp_l(7);
    Total_Result_Served_Local(1,counter_std) = temp_l(8);
    
    counter_std = counter_std + 1;
    std = std + std_step;
    
end
% save('D:\Simulation-MEC\NewCode-4-23-2020\STD-Analysis\Total_Result_satisfied_p_Happy.mat','Total_Result_satisfied_p_Happy');


%Total_Result
figure
% plot(10:length(Total_Result_satisfied_p_Random2), Total_Result_satisfied_p_Random2(1:total_run+1,1)*100,'DisplayName','Random');
% hold on
plot(start_std:std_step:std_max, Total_Result_satisfied_p_Greedy(1,1:counter_std -1)*100, '-o','DisplayName','GUS');
hold on
plot(start_std:std_step:std_max, Total_Result_satisfied_p_Happy(1,1:counter_std -1)*100,'-x','DisplayName','Happy Communication');
hold on
plot(start_std:std_step:std_max, Total_Result_satisfied_p_Happy_Comp(1,1:counter_std -1)*100,'->','DisplayName','Happy Computation');
hold on
plot(start_std:std_step:std_max, Total_Result_satisfied_p_Random(1,1:counter_std -1)*100,'-*','DisplayName','Random');
hold on
plot(start_std:std_step:std_max, Total_Result_satisfied_p_OffloadAll(1,1:counter_std -1)*100,'-^','DisplayName','Offload All');
hold on
plot(start_std:std_step:std_max, Total_Result_satisfied_p_Local(1,1:counter_std -1)*100,'-v','DisplayName','Local All');
hold on
%title('Evolution of User Satisfied Percent')
xlabel(X_label)
ylabel('User Satisfied Percent')
hold off
legend
set(legend,'location','best')
set(legend, 'Fontsize', 11)
ytickformat('percentage')
set(gca,'FontSize',18);

figure
% plot(start_std:100:std_max, Total_Result_satisfied_f_Random2(n,1:counter_std -1), '-s','DisplayName','RGO');
% hold on
plot(start_std:std_step:std_max, Total_Result_satisfied_f_Greedy(1,1:counter_std -1), '-o','DisplayName','GUS');
hold on
plot(start_std:std_step:std_max, Total_Result_satisfied_f_Happy(1,1:counter_std -1),'-x','DisplayName','Happy Communication');
hold on
plot(start_std:std_step:std_max, Total_Result_satisfied_f_Happy_Comp(1,1:counter_std -1),'->','DisplayName','Happy Computation');
hold on
plot(start_std:std_step:std_max, Total_Result_satisfied_f_Random(1,1:counter_std -1),'-*','DisplayName','Random');
hold on
plot(start_std:std_step:std_max, Total_Result_satisfied_f_OffloadAll(1,1:counter_std -1),'-^','DisplayName','Offload All');
hold on
plot(start_std:std_step:std_max, Total_Result_satisfied_f_Local(1,1:counter_std -1),'-v','DisplayName','Local All');
hold on
%title('Evolution of User Satisfaction')
xlabel(X_label)
ylabel('User Satisfaction Function Result')
hold off
legend
set(legend,'location','best')
set(legend, 'Fontsize', 11)
ytickformat('percentage')
set(gca,'FontSize',18);

figure
% plot(start_std:100:std_max, Total_Result_loss_Random2(n,1:counter_std -1) * 100, '-s','DisplayName','RGO');
% hold on
plot(start_std:std_step:std_max, Total_Result_loss_Greedy(1,1:counter_std -1) * 100, '-o','DisplayName','GUS');
hold on
plot(start_std:std_step:std_max, Total_Result_loss_Happy(1,1:counter_std -1) * 100,'-x','DisplayName','Happy Communication');
hold on
plot(start_std:std_step:std_max, Total_Result_loss_Happy_Comp(1,1:counter_std -1) * 100,'->','DisplayName','Happy Computation');
hold on
plot(start_std:std_step:std_max, Total_Result_loss_Random(1,1:counter_std -1) *100,'-*','DisplayName','Random');
hold on
plot(start_std:std_step:std_max, Total_Result_loss_OffloadAll(1,1:counter_std -1)*100,'-^','DisplayName','Offload All');
hold on
plot(start_std:std_step:std_max, Total_Result_loss_Local(1,1:counter_std -1)*100,'-v','DisplayName','Local All');
hold on
%title('Evolution of User Loss Percent')
xlabel(X_label)
ylabel('User Loss Percent')
hold off
legend
set(legend,'location','best')
set(legend, 'Fontsize', 11)
ytickformat('percentage')
set(gca,'FontSize',18);

figure
% plot(start_std:100:std_max, Total_Result_offloaded_Random2(n,1:counter_std -1)*100, '-s','DisplayName','RGO');
% hold on
plot(start_std:std_step:std_max, Total_Result_offloaded_Greedy(1,1:counter_std -1)*100, '-o','DisplayName','GUS');
hold on
plot(start_std:std_step:std_max, Total_Result_offloaded_Happy(1,1:counter_std -1)*100,'-x','DisplayName','Happy Communication');
hold on
plot(start_std:std_step:std_max, Total_Result_offloaded_Happy_Comp(1,1:counter_std -1)*100,'->','DisplayName','Happy Computation');
hold on
plot(start_std:std_step:std_max, Total_Result_offloaded_Random(1,1:counter_std -1)*100,'-*','DisplayName','Random');
hold on
plot(start_std:std_step:std_max, Total_Result_offloaded_OffloadAll(1,1:counter_std -1)*100,'-^','DisplayName','Offload All');
hold on
plot(start_std:std_step:std_max, Total_Result_offloaded_Local(1,1:counter_std -1)*100,'-v','DisplayName','Local All');
hold on
%title('Evolution of Offloaded Requests to Cloud')
xlabel(X_label)
ylabel('Offloaed Requests to Cloud Percent')
hold off
legend
set(legend,'location','best')
set(legend, 'Fontsize', 11)
ytickformat('percentage')
set(gca,'FontSize',18);

figure
% plot(start_std:100:std_max, Total_Result_local_Random2(n,1:counter_std -1)*100, '-s','DisplayName','RGO');
% hold on
plot(start_std:std_step:std_max, Total_Result_local_Greedy(1,1:counter_std -1)*100, '-o','DisplayName','GUS');
hold on
plot(start_std:std_step:std_max, Total_Result_local_Happy(1,1:counter_std -1)*100,'-x','DisplayName','Happy Communication');
hold on
plot(start_std:std_step:std_max, Total_Result_local_Happy_Comp(1,1:counter_std -1)*100,'->','DisplayName','Happy Computation');
hold on
plot(start_std:std_step:std_max, Total_Result_local_Random(1,1:counter_std -1)*100,'-*','DisplayName','Random');
hold on
plot(start_std:std_step:std_max, Total_Result_local_OffloadAll(1,1:counter_std -1)*100,'-^','DisplayName','Offload All');
hold on
plot(start_std:std_step:std_max, Total_Result_local_Local(1,1:counter_std -1)*100,'-v','DisplayName','Local All');
hold on
%title('Evolution of Local Processced Requests')
xlabel(X_label)
ylabel('Local Processced Requests Percent')
hold off
legend
set(legend,'location','best')
set(legend, 'Fontsize', 11)
ytickformat('percentage')
set(gca,'FontSize',18);

figure
% plot(start_std:100:std_max, Total_Result_dropped_Random2(n,1:counter_std -1)*100,'-s','DisplayName','RGO');
% hold on
plot(start_std:std_step:std_max, Total_Result_dropped_Greedy(1,1:counter_std -1)*100, '-o','DisplayName','GUS');
hold on
plot(start_std:std_step:std_max, Total_Result_dropped_Happy(1,1:counter_std -1)*100,'-x','DisplayName','Happy Communication');
hold on
plot(start_std:std_step:std_max, Total_Result_dropped_Happy_Comp(1,1:counter_std -1)*100,'->','DisplayName','Happy Computation');
hold on
plot(start_std:std_step:std_max, Total_Result_dropped_Random(1,1:counter_std -1)*100,'-*','DisplayName','Random');
hold on
plot(start_std:std_step:std_max, Total_Result_dropped_OffloadAll(1,1:counter_std -1)*100,'-^','DisplayName','Offload All');
hold on
plot(start_std:std_step:std_max, Total_Result_dropped_Local(1,1:counter_std -1)*100,'-v','DisplayName','Local All');
hold on
%title('Evolution of Dropped Requests')
xlabel(X_label)
ylabel('Dropped Requests Percent')
hold off
legend
set(legend,'location','best')
set(legend, 'Fontsize', 11)
ytickformat('percentage')
set(gca,'FontSize',18);


figure
% plot(start_std:100:std_max, Total_Result_dropped_Random2(n,1:counter_std -1)*100,'-s','DisplayName','RGO');
% hold on
plot(start_std:std_step:std_max, Total_Result_offloaded_toEdge_Greedy(1,1:counter_std -1)*100, '-o','DisplayName','GUS');
hold on
plot(start_std:std_step:std_max, Total_Result_offloaded_toEdge_Happy(1,1:counter_std -1)*100,'-x','DisplayName','Happy Communication');
hold on
plot(start_std:std_step:std_max, Total_Result_offloaded_toEdge_Happy_Comp(1,1:counter_std -1)*100,'->','DisplayName','Happy Computation');
hold on
plot(start_std:std_step:std_max, Total_Result_offloaded_toEdge_Random(1,1:counter_std -1)*100,'-*','DisplayName','Random');
hold on
plot(start_std:std_step:std_max, Total_Result_offloaded_toEdge_OffloadAll(1,1:counter_std -1)*100,'-^','DisplayName','Offload All');
hold on
plot(start_std:std_step:std_max, Total_Result_offloaded_toEdge_Local(1,1:counter_std -1)*100,'-v','DisplayName','Local All');
hold on
%title('Evolution of Offloaded to Requests Neighborhood Edges')
xlabel(X_label)
ylabel('Offloaded Requests to Neighborhood Edge Percent')
hold off
legend
set(legend,'location','best')
set(legend, 'Fontsize', 11)
ytickformat('percentage')
set(gca,'FontSize',18);

figure
plot(start_std:std_step:std_max, Total_Result_Accuracy_Greedy(1,1:counter_std -1), '-o','DisplayName','w1*Accurcay');
hold on
plot(start_std:std_step:std_max, Total_Result_Delay_Greedy(1,1:counter_std -1),'-x','DisplayName','w2*Delay');
hold on
%title('Evolution of Accurcay and Delay')
xlabel(X_label)
ylabel('Value')
hold off
legend
set(legend,'location','best')
set(legend, 'Fontsize', 11)
ytickformat('percentage')
set(gca,'FontSize',18);

figure
% plot(start_std:100:std_max, Total_Result_dropped_Random2(n,1:counter_std -1)*100,'-s','DisplayName','RGO');
% hold on
plot(start_std:std_step:std_max, Total_Result_Served_Greedy(1,1:counter_std -1)*100, '-o','DisplayName','GUS');
hold on
plot(start_std:std_step:std_max, Total_Result_Served_Happy(1,1:counter_std -1)*100,'-x','DisplayName','Happy Communication');
hold on
plot(start_std:std_step:std_max, Total_Result_Served_Happy_Comp(1,1:counter_std -1)*100,'->','DisplayName','Happy Computation');
hold on
plot(start_std:std_step:std_max, Total_Result_Served_Random(1,1:counter_std -1)*100,'-*','DisplayName','Random');
hold on
plot(start_std:std_step:std_max, Total_Result_Served_OffloadAll(1,1:counter_std -1)*100,'-^','DisplayName','Offload All');
hold on
plot(start_std:std_step:std_max, Total_Result_Served_Local(1,1:counter_std -1)*100,'-v','DisplayName','Local All');
hold on
%title('Evolution of Served Requests')
xlabel(X_label)
ylabel('Served Requests Percent')
hold off
legend
set(legend,'location','best')
set(legend, 'Fontsize', 11)
ytickformat('percentage')
set(gca,'FontSize',18);
