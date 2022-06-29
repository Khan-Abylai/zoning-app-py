from app import App

if __name__ == '__main__':
    config_file = "/home/user/parking_zoning/dev/config.json"
    application = App(config_file=config_file)
    application.run()
