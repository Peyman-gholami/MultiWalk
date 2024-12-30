#%%
import re
import os
from datetime import datetime
import numpy as np

# Define the config dictionary specifying the required attributes for each event type


config = {
    'communication': ['end_time', 'bytes_sent', 'duration'],
    'local sgd': ['end_time', 'duration', 'iteration', 'epoch'],
    'evaluation': ['start_time', 'train loss', 'test loss']
}

duration = 30 * 60
local_steps_per_iter = 5

def check_criteria(log_file, criteria):
    # Open the log file for reading
    # print(criteria)
    with open(log_file, 'r') as file:
        for line in file:
            # Check if the line starts with "INFO:"
            if line.startswith("INFO:./log/"):
                # Find the position of the first colon after "INFO:"
                first_colon_index = line.find(":", len("INFO:"))

                # Extract the directory path (after "INFO:" and before the first colon)
                directory_path = line[len("INFO:"):first_colon_index]
                # print(directory_path)

                if all(criterion in directory_path for criterion in criteria):
                    return True
                return False

                # Extract the log information (everything after the first colon)
                log_info = line[first_colon_index + 1:].strip()

                # Create directories if they do not exist
                directory, filename = os.path.split(directory_path)
                if not os.path.exists(directory):
                    os.makedirs(directory)

                # Determine the full file path to write the log
                full_file_path = os.path.join(directory, filename)

                # Open the corresponding log file in append mode and write the log information
                with open(full_file_path, 'a') as log_file:
                    log_file.write(log_info + '\n')

def parse_line(line):
    """Parse a single log line and extract the required information."""
    event_type = line.split(',')[0]
    attributes = dict(re.findall(r'(\w+): ([^,]+)', line))
    extracted_values = []
    if event_type in config:
        for key in config[event_type]:
            if event_type == 'evaluation' and key in ['test loss', 'train loss']:
                loss_info = re.search(f"{key}: {{'loss': ([^,]+), 'accuracy': ([^}}]+)}}", line)
                if loss_info:
                    extracted_values.extend([float(loss_info.group(1)), float(loss_info.group(2))])
            else:
                value = attributes[key].strip('} \n')
                if 'time' in key:
                    extracted_values.append(value)
                elif key == 'duration':
                    extracted_values.append(float(value.rstrip('s')))
                else:
                    extracted_values.append(float(value))
    return event_type, extracted_values


def parse_log_file(filename):
    """Parse a log file and return the extracted data."""
    parsed_data = {
        'communication': [],
        'local sgd': [],
        'evaluation': []
    }
    with open(filename, 'r') as file:
        for line in file:
            # Check if the line starts with "INFO:./log/"
            if line.startswith("INFO:./log/"):
                first_colon_index = line.find(":", len("INFO:"))
                # Extract the log information (everything after the log file name)
                log_info = line[first_colon_index + 1:].strip()
                event_type, extracted_values = parse_line(log_info)
                parsed_data[event_type].append(extracted_values)
    return parsed_data


def convert_to_datetime(time_str):
    """Convert a time string to a datetime object."""
    return datetime.strptime(time_str, '%Y-%m-%d %H:%M:%S %Z')


def aggregate_logs(base_path, criteria):
    """Aggregate log data from multiple files matching all specified criteria."""
    all_parsed_data = []

    # List all files in the base path directory
    print(criteria)
    for file in os.listdir(base_path):
        log_filename = os.path.join(base_path, file)
        if check_criteria(log_filename, criteria):
            parsed_data = parse_log_file(log_filename)
            print(log_filename)


            # Find start and end time from computation
            computation_times = [convert_to_datetime(entry[0]) for entry in parsed_data['local sgd']]
            start_time = min(computation_times)
            end_time = max(computation_times)
            # print(start_time, end_time)
            # Filter and normalize times, and convert to seconds since start_time
            for event in parsed_data:
                filtered_entries = []
                for entry in parsed_data[event]:
                    event_time = convert_to_datetime(entry[0])
                    if start_time <= event_time <= end_time:
                        # Normalize time
                        normalized_time = (event_time - start_time).total_seconds()

                        # Replace time with normalized time
                        filtered_entries.append([normalized_time] + entry[1:])
                parsed_data[event] = filtered_entries

            # Sort entries by normalized time
            for event in parsed_data:
                parsed_data[event].sort(key=lambda x: x[0])

            # Cumulative aggregation for communication and local sgd
            def cumulative_aggregate(entries):
                """Generate cumulative aggregates for entries."""
                cumulative_entries = []
                # Initialize the current aggregate, keeping time separate
                current_aggregate = [entries[0][0]] + [0] * (len(entries[0]) - 1)
                for entry in entries:
                    # Update cumulative values except the first element (time)
                    current_aggregate = [entry[0]] + [current_aggregate[i] + entry[i] for i in range(1, len(entry))]
                    cumulative_entries.append(current_aggregate.copy())
                return cumulative_entries

            for event in ['communication', 'local sgd']:
                if parsed_data[event]:
                    cumulative_data = cumulative_aggregate(parsed_data[event])
                    parsed_data[event] = cumulative_data

            all_parsed_data.append(parsed_data)

    return all_parsed_data


# Define the base path and criteria
base_path = '/Users/peyman_gh/.kube/noniid_logs'
criterias = [
    ["rw=1graph=erdos_renyi",
     "learning_rate=0.05",
     "algorithm=random_walk",
     "task=Cifar",
     "split_method=dirichlet_non_iid_alpha=0.1"
     ],
    ["rw=4graph=erdos_renyi",
    "learning_rate=0.05",
    "algorithm=random_walk",
    "task=Cifar",
     "split_method=dirichlet_non_iid_alpha=0.1"
     ],
    ["rw=8graph=erdos_renyi",
     "learning_rate=0.05",
     "algorithm=random_walk",
     "task=Cifar",
     "split_method=dirichlet_non_iid_alpha=0.1"
     ],
    ["rw=12graph=erdos_renyi",
     "learning_rate=0.05",
     "algorithm=random_walk",
     "task=Cifar",
     "split_method=dirichlet_non_iid_alpha=0.1"],
    # #
    # ["rw=15graph=cycle",
    #  "learning_rate=0.05",
    #  "algorithm=random_walk",
    #  "task=Cifar",
    #  "split_method=dirichlet_non_iid_alpha=1.0"],
    # ["rw=20graph=complete",
    #  "learning_rate=0.05",
    #  "algorithm=random_walk",
    #  "task=Cifar",
    #  "split_method=dirichlet_non_iid_alpha=1"],
    # #
    # ["rw=6graph=cycle",
    #  "learning_rate=0.05",
    #  "algorithm=random_walk",
    #  "task=Cifar",
    #  "split_method=dirichlet_non_iid_alpha=10"],
    #
    # ["rw=12learning",
    #  "learning_rate=0.01",
    #  "algorithm=random_walk",
    #  "task=Cifar",
    #  "split_method=dirichlet_non_iid_alpha=0.1"],



    #
    # ["rw=4learning",
    # "learning_rate=0.05",
    # "algorithm=random_walk",
    # "task=Cifar"],

    # ["rw=6learning",
    # "learning_rate=0.01",
    # "algorithm=random_walk",
    # "task=Cifar",
    # "split_method=dirichlet_non_iid_alpha=1.0"],

    # ["rw=12learning",
    #  "learning_rate=0.01",
    #  "algorithm=random_walk",
    #  "task=Cifar",
    #  "split_method=dirichlet_non_iid_alpha=1.0"],



    # ["size=20_rank=0_rw=15l", "algorithm=random"],

    # ["learning_rate=0.05",
    # "algorithm=async_gossip_task",
    # "task=Cifar"],
    #
    # ["full_dup_size=20_",
    #  "algorithm=async_gossip_task",],
    #
    # ["full_dup_size=20_rank=0_rw=14"],

    ["algorithm=async_gossip_task",
     "graph=erdos_renyi",
     "learning_rate=0.05",
     "task=Cifar",
     "split_method=dirichlet_non_iid_alpha=0.1"
     ]

]

# Aggregate logs for the specified criteria
data=[]

for criteria in criterias:
    aggregated_data = aggregate_logs(base_path, criteria)
    data.append(aggregated_data)

#%%
t_values = np.linspace(0,duration,50)#np.linspace(0,min(min(data[i][j]["evaluation"][-1] for j in range(len(data[i]))) for trial in trials), 100)
processed_mean = []
processed_std = []

for case in data:
    processsd_mean_case = {}
    processsd_std_case = {}
    for key in case[0]:
        print(key)
        y_count = len(case[0][key][0]) - 1
        interpolated_y_values = [[] for _ in range(y_count)]

        for trial in case:
            trial = trial[key]
            t_trial = [x[0] for x in trial]
            y_trials = [list(y[i+1] for y in trial) for i in range(y_count)]
            for i in range(y_count):
                y_interp = np.interp(t_values, t_trial, y_trials[i])
                interpolated_y_values[i].append(y_interp)

        interpolated_y_values = [np.array(y) for y in interpolated_y_values]
        mean_y = [np.mean(y_values, axis=0) for y_values in interpolated_y_values]
        std_y = [np.std(y_values, axis=0) for y_values in interpolated_y_values]
        processsd_mean_case[key] = mean_y
        processsd_std_case[key] = std_y

    processed_mean.append(processsd_mean_case)
    processed_std.append(processsd_std_case)



#%%
import matplotlib.pyplot as plt
import matplotlib

x_values_list = [
    [t_values for i in range(len(data))],
    [processed_mean[i]['local sgd'][1]/100/local_steps_per_iter for i in range(len(data))],
    [processed_mean[i]['communication'][0] / 1024 / 1024 / 1024 for i in range(len(data))]
]
output_filenames_list = ["figure_time_", "figure_iters_", "figure_communication_"]
xlabel_list = ["time (seconds)", "iteration ($10^2$)", "communication overhead (GB)"]

for idx in range(len(x_values_list)):

    fig,ax = plt.subplots(figsize=(24,10),nrows=2, ncols=2)
    ax = ax.flatten()
    plt.rcParams.update({'font.size': 19})
    plt.xticks(fontsize = 19)
    plt.yticks(fontsize = 19)
    colors = matplotlib.cm.tab20(range(20))
    b=0
    c=0
    markers=["o","X","P","^","v","s","h","<",">","d","*"]
    every=[5,5,5,5,5,5,5,5,5,5,5]
    order = [10,0,9,5,5,6,5,6,6]
    # legends = ["MW-4","MW-4", "MW-8", "MW-12", "AD-PSGD"]
    legends = ["MW-1", "MW-4","MW-8","MW-12", "AD-PSGD"]


    for i in range(len(data)):
        x = x_values_list[idx][i]
        y = processed_mean[i]['evaluation'][0]
        ax[0].plot(x, y)
        std = processed_std[i]['evaluation'][0]
        ax[0].fill_between(x, y - std, y + std, alpha=0.2,label='_nolegend_')

    for i,line in enumerate(ax[0].get_lines()):
        # line.set_marker(markers[i])
        # line.set_markevery(5)
        # line.set_color(colors[i])
        line.set_markersize(10)

    ax[0].set_ylabel('training global loss')
    ax[0].set_xlabel(xlabel_list[idx]) #ax[0].set_xlabel('time (seconds)')
    ax[0].legend(legends)
    ax[0].grid(True,which="both")
    ax[0].set_ylim([0, 4])



    for i in range(len(data)):
        x = x_values_list[idx][i]
        y = processed_mean[i]['evaluation'][1]#[entry[2] for entry in processed_mean[i]['evaluation']]
        ax[1].plot(x, y)
        std = processed_std[i]['evaluation'][1]
        ax[1].fill_between(x, y - std, y + std, alpha=0.2, label='_nolegend_')

    for i,line in enumerate(ax[1].get_lines()):
        # line.set_marker(markers[i])
        # line.set_markevery(5)
        # line.set_color(colors[i])
        line.set_markersize(10)

    ax[1].set_ylabel('training accuracy')
    ax[1].set_xlabel(xlabel_list[idx])#ax[1].set_xlabel('iteration')#ax[1].set_xlabel('time (seconds)')
    ax[1].legend(legends)
    ax[1].grid(True,which="both")




    for i in range(len(data)):
        x = x_values_list[idx][i]
        y = processed_mean[i]['communication'][0]/ 1024 / 1024 / 1024 #[entry[1] / 1024 / 1024 for entry in data[i]['communication']]
        ax[2].plot(x, y)
        std = processed_std[i]['communication'][0]/ 1024 / 1024 / 1024
        ax[2].fill_between(x, y - std, y + std, alpha=0.2, label='_nolegend_')

    for i,line in enumerate(ax[2].get_lines()):
        # line.set_marker(markers[i])
        # line.set_markevery(50)
        # line.set_color(colors[i])
        line.set_markersize(5)

    ax[2].set_ylabel('communication overhead (GB)')
    ax[2].set_xlabel(xlabel_list[idx])#ax[2].set_xlabel('time (seconds)')
    ax[2].legend(legends)
    ax[2].set_yscale('log')
    ax[2].grid(True,which="both")



    for i in range(len(data)):
        x = x_values_list[idx][i]
        y = processed_mean[i]['local sgd'][0]#[entry[1] / 60 for entry in data[i]['local sgd']]
        ax[3].plot(x, y)
        std = processed_std[i]['local sgd'][0]
        ax[3].fill_between(x, y - std, y + std, alpha=0.2, label='_nolegend_')

    for i,line in enumerate(ax[3].get_lines()):
        # line.set_marker(markers[i])
        # line.set_markevery(50)
        # line.set_color(colors[i])
        line.set_markersize(5)

    ax[3].set_ylabel('GPU usage (minutes)')
    ax[3].set_xlabel(xlabel_list[idx])#ax[3].set_xlabel('time (seconds)')
    ax[3].legend(legends)
    # ax[3].set_yscale('log')
    ax[3].grid(True,which="both")

    plt.tight_layout()

    # Save the figure as a PDF
    plt.savefig("./figures/"+output_filenames_list[idx]+" ".join(criterias[0])+".pdf", format='pdf', bbox_inches='tight')
    plt.show()