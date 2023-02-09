from sqlalchemy.orm import declarative_base
from sqlalchemy import create_engine, Column, Integer, Float

Base = declarative_base()


class Position(Base):
    __tablename__ = 'positions'

    trajectory_id = Column(Integer, primary_key=True)
    frame_number = Column(Integer, primary_key=True)
    x_coordinate = Column(Float)
    y_coordinate = Column(Float)
    line_n = Column(Integer)


class Velocity(Base):
    __tablename__ = 'velocities'

    id = Column(Integer, primary_key=True)
    trajectory_id = Column(Integer)
    frame_number = Column(Integer)
    x_coordinate = Column(Float)
    y_coordinate = Column(Float)


class Object(Base):
    __tablename__ = 'objects'
    object_id = Column(Integer, primary_key=True)
    road_user_type = Column(Integer)


if __name__ == '__main__':
    engine = create_engine('sqlite:///data.db')
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
