import hometrainer.util


def main():
    with open('work_dir/winrate.png', 'wb') as file:
        plot = hometrainer.util.plot_external_eval_avg_score('work_dir', 0, -1, True, smoothing=0.15)
        file.write(plot.getbuffer())


if __name__ == '__main__':
    main()
