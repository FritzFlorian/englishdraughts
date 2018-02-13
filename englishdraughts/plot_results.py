import hometrainer.util


def main():
    with open('new_run/winrate.png', 'wb') as file:
        plot = hometrainer.util.plot_external_eval_avg_score('new_run', 0, -1, False, smoothing=0.15)
        file.write(plot.getbuffer())


if __name__ == '__main__':
    main()
