out port p;

int multi_tile_main(){

    chanend c, d, e, f;

    on tile[0]: {
        select Foo_impl
    };
    on tile[1]: {

    }
    return 0;
}


////////////////////////////////////////////

/*
ordered
select_object Foo(port a, port b, timer t, port c){
    a_guard() => a :> int: a_handler();
    b_guard() => b :> int: b_handler();
}
*/

typedef struct {
    out port a;
    in port b;
    timer t;
    int a_enabled;
} Foo;

int prolog(const Foo *state, out port a, in port b, timer t);//??

int a_guard(const Foo *state){ return state->a_enabled;}
void a_handler(const Foo *state, int data);
int b_guard(const Foo *state);
void b_handler(const Foo *state, int data);
int t_guard(const Foo *state);
void t_handler(const Foo *state, int data);

